use num_traits::Zero;
use std::{borrow::Cow, marker::PhantomData, path::Path};

use super::{Backwards, Forwards, OneWayTransform};
use crate::{
    filter::{backwards_window, forwards_window, upscale_window, Filter},
    stream::{CompressionLevel, Deserializable, DeserializeStream, Serializable, SerializeStream},
    volume::{VolumeBlock, VolumeWindowMut},
};

/// Implementation of a wavelet transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaveletTransform<T: Send, F: Filter<T>> {
    filter: F,
    split_high: bool,
    _phantom: PhantomData<fn() -> T>,
}

impl<T, F> WaveletTransform<T, F>
where
    T: Zero + Clone + Send,
    F: Filter<T>,
{
    /// Constructs a new `WaveletTransform` with the provided filter.
    pub fn new(filter: F, split_high: bool) -> Self {
        Self {
            filter,
            split_high,
            _phantom: PhantomData,
        }
    }

    fn forw_(
        &self,
        mut input: VolumeWindowMut<'_, T>,
        mut scratch: Vec<T>,
        ops: &[ForwardsOperation],
    ) {
        if ops.is_empty() {
            return;
        }

        let dim = ops[0].dim;
        let has_high_pass = ops.get(1).map(|o| o.dim > dim).unwrap_or(false);
        let has_high_pass = has_high_pass && self.split_high;

        forwards_window(dim, &self.filter, &mut input, &mut scratch);

        let (input_low, input_high) = input.split_into(dim);
        rayon::scope(|s| {
            s.spawn(|_| self.forw_(input_low, scratch, &ops[1..]));

            if has_high_pass {
                s.spawn(|_| self.forw_high(input_high, None, &ops[1..], dim));
            }
        });
    }

    fn forw_high(
        &self,
        mut input: VolumeWindowMut<'_, T>,
        scratch: Option<Vec<T>>,
        ops: &[ForwardsOperation],
        last_dim: usize,
    ) {
        if ops.is_empty() || ops[0].dim < last_dim {
            return;
        }

        let dim = ops[0].dim;
        let mut scratch =
            scratch.unwrap_or_else(|| vec![T::zero(); *input.dims().iter().max().unwrap()]);
        forwards_window(dim, &self.filter, &mut input, &mut scratch);

        let (input_low, input_high) = input.split_into(dim);
        rayon::scope(|s| {
            s.spawn(|_| self.forw_high(input_low, Some(scratch), &ops[1..], dim));
            s.spawn(|_| self.forw_high(input_high, None, &ops[1..], dim));
        });
    }

    pub(crate) fn back_(
        &self,
        mut input: VolumeWindowMut<'_, T>,
        scratch: &mut [T],
        ops: &[BackwardsOperation],
    ) {
        if ops.is_empty() {
            return;
        }

        match &ops[0] {
            BackwardsOperation::Backwards { dim, adapt } => {
                let has_high_pass = ops.get(1).map(|o| o.dim() > *dim).unwrap_or(false);
                let has_high_pass = has_high_pass && self.split_high;

                let (low, high) = input.split_mut(*dim);

                let range = adapt.as_ref().map(|(range, _)| range);
                let steps = adapt.as_ref().map(|(_, steps)| &**steps);

                rayon::scope(|s| {
                    s.spawn(|_| self.back_(low, scratch, &ops[1..]));

                    if has_high_pass {
                        s.spawn(|_| self.back_high(high, &ops[1..], steps, *dim));
                    } else if let Some(steps) = steps {
                        let ops = ForwardsOperation::new(steps);
                        let scratch = vec![T::zero(); *high.dims().iter().max().unwrap()];
                        self.forw_(high, scratch, &ops);
                    }
                });

                let mut input = if let Some(range) = range {
                    input.custom_range_mut(range)
                } else {
                    input
                };
                backwards_window(*dim, &self.filter, &mut input, scratch);
            }
            BackwardsOperation::Skip { dim } => {
                let has_high_pass = ops.get(1).map(|o| o.dim() > *dim).unwrap_or(false);
                let has_high_pass = has_high_pass && self.split_high;

                let (low, high) = input.split_mut(*dim);

                rayon::scope(|s| {
                    s.spawn(|_| self.back_(low, scratch, &ops[1..]));

                    if has_high_pass {
                        s.spawn(|_| self.back_high(high, &ops[1..], None, *dim));
                    }
                });
            }
        }
    }

    fn back_high(
        &self,
        mut input: VolumeWindowMut<'_, T>,
        ops: &[BackwardsOperation],
        steps: Option<&[u32]>,
        last_dim: usize,
    ) {
        if ops.is_empty() {
            return;
        }

        match ops[0] {
            BackwardsOperation::Backwards { dim, .. } => {
                if dim < last_dim {
                    return;
                }

                let (low, high) = input.split_mut(dim);
                rayon::scope(|s| {
                    s.spawn(|_| self.back_high(low, &ops[1..], None, dim));
                    s.spawn(|_| self.back_high(high, &ops[1..], None, dim));
                });

                let mut scratch = vec![T::zero(); input.dims()[dim]];
                backwards_window(dim, &self.filter, &mut input, &mut scratch);
                drop(scratch);
            }
            BackwardsOperation::Skip { dim } => {
                if dim < last_dim {
                    return;
                }

                let (low, high) = input.split_mut(dim);
                rayon::scope(|s| {
                    s.spawn(|_| self.back_high(low, &ops[1..], None, dim));
                    s.spawn(|_| self.back_high(high, &ops[1..], None, dim));
                });

                upscale_window(dim, &mut input);
            }
        }

        if let Some(steps) = steps {
            let ops = ForwardsOperation::new(steps);
            let scratch = vec![T::zero(); *input.dims().iter().max().unwrap()];
            self.forw_(input, scratch, &ops);
        }
    }
}

impl<T, F> OneWayTransform<Forwards, VolumeBlock<T>> for WaveletTransform<T, F>
where
    T: Zero + Clone + Send,
    F: Filter<T>,
{
    type Result = VolumeBlock<T>;
    type Cfg<'a> = WaveletDecompCfg<'a>;

    fn apply(&self, mut input: VolumeBlock<T>, cfg: Self::Cfg<'_>) -> VolumeBlock<T> {
        let dims = input.dims();

        assert!(dims.len() == cfg.steps.len());
        for (dim, step) in dims.iter().zip(cfg.steps.iter()) {
            let required_size = 2usize.pow(*step);
            assert!(required_size <= *dim);
        }

        if dims.iter().cloned().product::<usize>() == 0 {
            panic!("invalid number of steps");
        }

        if dims.iter().cloned().product::<usize>() == 1 {
            return input;
        }

        let scratch = vec![T::zero(); *input.dims().iter().max().unwrap()];
        let ops = ForwardsOperation::new(cfg.steps);
        let input_window = input.window_mut();

        self.forw_(input_window, scratch, &ops);

        input
    }
}

impl<T, F> OneWayTransform<Backwards, VolumeBlock<T>> for WaveletTransform<T, F>
where
    T: Zero + Clone + Send,
    F: Filter<T>,
{
    type Result = VolumeBlock<T>;
    type Cfg<'a> = WaveletRecompCfg<'a>;

    fn apply(&self, mut input: VolumeBlock<T>, cfg: Self::Cfg<'_>) -> VolumeBlock<T> {
        let dims = input.dims();

        assert!(dims.len() == cfg.steps.len());
        for (&dim, &b) in dims.iter().zip(cfg.steps) {
            let required_size = 2usize.pow(b);
            assert!(required_size <= dim);
        }

        if dims.iter().cloned().product::<usize>() == 0 {
            panic!("invalid number of steps");
        }

        if dims.iter().cloned().product::<usize>() == 1 || cfg.steps.iter().all(|&f| f == 0) {
            return input;
        }

        let mut scratch = vec![T::zero(); *input.dims().iter().max().unwrap()];
        let (ops, output_size) =
            BackwardsOperation::new(cfg.steps, &*cfg.decomposition, input.dims());
        let input_window = input.window_mut();

        self.back_(input_window, &mut scratch, &ops);

        let mut input_window = input.window_mut();
        if cfg.upscale {
            for (dim, &size) in output_size.iter().enumerate() {
                let upscale_steps = (input_window.dims()[dim] / size).ilog2();
                for _ in 0..upscale_steps {
                    upscale_window(dim, &mut input_window);
                }
            }
            input
        } else {
            let input_window = input_window.custom_range(&output_size);
            input_window.as_block()
        }
    }
}

impl<T, F> Serializable for WaveletTransform<T, F>
where
    T: Send,
    F: Filter<T> + Serializable,
{
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        std::any::type_name::<T>().serialize(stream);
        std::any::type_name::<F>().serialize(stream);
        self.filter.serialize(stream);
        self.split_high.serialize(stream);
    }
}

impl<T, F> Deserializable for WaveletTransform<T, F>
where
    T: Send,
    F: Filter<T> + Deserializable,
{
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let n_type: String = Deserializable::deserialize(stream);
        let t_type: String = Deserializable::deserialize(stream);
        assert_eq!(n_type, std::any::type_name::<T>());
        assert_eq!(t_type, std::any::type_name::<F>());

        let filter = Deserializable::deserialize(stream);
        let split_high = Deserializable::deserialize(stream);
        Self {
            filter,
            split_high,
            _phantom: PhantomData,
        }
    }
}

/// Config for decomposing a volume with the wavelet transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaveletDecompCfg<'a> {
    steps: &'a [u32],
}

impl<'a> WaveletDecompCfg<'a> {
    /// Constructs a new `WaveletDecompCfg`.
    pub fn new(steps: &'a [u32]) -> Self {
        Self { steps }
    }

    /// Returns the number of decomposition steps.
    pub fn steps(&self) -> &'a [u32] {
        self.steps
    }
}

impl<'a> From<&'a WaveletDecompCfgOwned> for WaveletDecompCfg<'a> {
    fn from(x: &'a WaveletDecompCfgOwned) -> Self {
        Self { steps: &x.steps }
    }
}

impl Serializable for WaveletDecompCfg<'_> {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.steps.serialize(stream)
    }
}

/// Config for recomposing a volume with the wavelet transform.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaveletRecompCfg<'a> {
    steps: &'a [u32],
    decomposition: Cow<'a, [usize]>,
    upscale: bool,
}

impl<'a> WaveletRecompCfg<'a> {
    /// Constructs a new `WaveletRecompCfg`.
    pub fn new(steps: &'a [u32], decomposition: Option<&'a [usize]>) -> Self {
        Self::new_with_upscale(steps, decomposition, true)
    }

    /// Constructs a new `WaveletRecompCfg`.
    pub fn new_with_upscale(
        steps: &'a [u32],
        decomposition: Option<&'a [usize]>,
        upscale: bool,
    ) -> Self {
        if let Some(decomp) = decomposition {
            Self {
                steps,
                decomposition: Cow::Borrowed(decomp),
                upscale,
            }
        } else {
            let mut remaining_steps = Vec::from(steps);
            let mut decomp = vec![];

            while remaining_steps.iter().any(|&s| s != 0) {
                for (i, s) in remaining_steps.iter_mut().enumerate() {
                    if *s > 0 {
                        *s -= 1;
                        decomp.push(i);
                    }
                }
            }

            Self {
                steps,
                decomposition: Cow::Owned(decomp),
                upscale,
            }
        }
    }
}

impl<'a> From<WaveletDecompCfg<'a>> for WaveletRecompCfg<'a> {
    fn from(x: WaveletDecompCfg<'a>) -> Self {
        Self::new(x.steps, None)
    }
}

impl<'a> From<&'a WaveletRecompCfgOwned> for WaveletRecompCfg<'a> {
    fn from(x: &'a WaveletRecompCfgOwned) -> Self {
        Self {
            steps: &x.steps,
            decomposition: Cow::Borrowed(&x.decomposition),
            upscale: x.upscale,
        }
    }
}

impl Serializable for WaveletRecompCfg<'_> {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.steps.serialize(stream);
        self.decomposition.serialize(stream);
        self.upscale.serialize(stream);
    }
}

/// Owned variant of a [`WaveletDecompCfg`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaveletDecompCfgOwned {
    steps: Vec<u32>,
}

impl WaveletDecompCfgOwned {
    /// Constructs a new `WaveletDecompCfgOwned`.
    pub fn new(steps: Vec<u32>) -> Self {
        Self { steps }
    }
}

impl From<WaveletDecompCfg<'_>> for WaveletDecompCfgOwned {
    fn from(x: WaveletDecompCfg<'_>) -> Self {
        Self::new(x.steps.into())
    }
}

impl Serializable for WaveletDecompCfgOwned {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.steps.serialize(stream)
    }
}

impl Deserializable for WaveletDecompCfgOwned {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let steps = Deserializable::deserialize(stream);
        Self { steps }
    }
}

/// Owned variant of a [`WaveletRecompCfg`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaveletRecompCfgOwned {
    steps: Vec<u32>,
    decomposition: Vec<usize>,
    upscale: bool,
}

impl WaveletRecompCfgOwned {
    /// Constructs a new `WaveletRecompCfgOwned`.
    pub fn new(steps: Vec<u32>, decomposition: Option<Vec<usize>>) -> Self {
        Self::new_with_upsize(steps, decomposition, true)
    }

    /// Constructs a new `WaveletRecompCfgOwned`.
    pub fn new_with_upsize(
        steps: Vec<u32>,
        decomposition: Option<Vec<usize>>,
        upscale: bool,
    ) -> Self {
        if let Some(decomp) = decomposition {
            Self {
                steps,
                decomposition: decomp,
                upscale,
            }
        } else {
            let mut remaining_steps = steps.clone();
            let mut decomp = vec![];

            while remaining_steps.iter().any(|&s| s != 0) {
                for (i, s) in remaining_steps.iter_mut().enumerate() {
                    if *s > 0 {
                        *s -= 1;
                        decomp.push(i);
                    }
                }
            }

            Self {
                steps,
                decomposition: decomp,
                upscale,
            }
        }
    }
}

impl From<WaveletRecompCfg<'_>> for WaveletRecompCfgOwned {
    fn from(x: WaveletRecompCfg<'_>) -> Self {
        Self::new_with_upsize(x.steps.into(), Some(x.decomposition.to_vec()), x.upscale)
    }
}

impl Serializable for WaveletRecompCfgOwned {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.steps.serialize(stream);
        self.decomposition.serialize(stream);
        self.upscale.serialize(stream);
    }
}

impl Deserializable for WaveletRecompCfgOwned {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let steps = Deserializable::deserialize(stream);
        let decomposition = Deserializable::deserialize(stream);
        let upscale = Deserializable::deserialize(stream);

        Self {
            steps,
            decomposition,
            upscale,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ForwardsOperation {
    pub dim: usize,
}

impl ForwardsOperation {
    pub fn new(steps: &[u32]) -> Vec<Self> {
        let mut ops = Vec::new();
        let mut step = vec![0; steps.len()];

        let mut stop = false;
        while !stop {
            stop = true;

            for (i, (step, max)) in step.iter_mut().zip(steps).enumerate() {
                if *step < *max {
                    *step += 1;
                    stop = false;
                    ops.push(Self { dim: i });
                }
            }
        }

        ops
    }
}

#[derive(Debug, Clone)]
pub(crate) enum BackwardsOperation {
    Backwards {
        dim: usize,
        adapt: Option<(Vec<usize>, Vec<u32>)>,
    },
    Skip {
        dim: usize,
    },
}

impl BackwardsOperation {
    fn new(steps: &[u32], decomposition: &[usize], dims: &[usize]) -> (Vec<Self>, Vec<usize>) {
        assert_eq!(steps.len(), dims.len());

        let mut dims = Vec::from(dims);
        for &dim in decomposition {
            dims[dim] /= 2;
        }

        let mut steps = Vec::from(steps);
        let mut skipped = vec![0u32; steps.len()];
        let ops = decomposition
            .iter()
            .rev()
            .map(|&dim| {
                if steps[dim] == 0 {
                    skipped[dim] += 1;
                    return BackwardsOperation::Skip { dim };
                }
                dims[dim] *= 2;
                steps[dim] -= 1;

                let adapt = if skipped.iter().all(|&s| s == 0) {
                    None
                } else {
                    Some((dims.clone(), skipped.clone()))
                };

                BackwardsOperation::Backwards { dim, adapt }
            })
            .collect::<Vec<_>>();

        let ops = ops.into_iter().rev().collect();
        (ops, dims)
    }

    fn dim(&self) -> usize {
        match self {
            BackwardsOperation::Backwards { dim, .. } => *dim,
            BackwardsOperation::Skip { dim } => *dim,
        }
    }
}

pub(crate) fn write_out_block<T>(
    path: impl AsRef<Path>,
    block: VolumeBlock<T>,
    compression: CompressionLevel,
) where
    T: Serializable + Clone,
{
    let path = path.as_ref();
    if !path.exists() || !path.is_dir() {
        panic!("Invalid path {path:?}");
    }

    let dims = Vec::from(block.dims());
    let mut max_steps = vec![0u32; dims.len()];
    let mut decomposition = Vec::new();

    let mut part_idx = 0;
    let mut window = block.window();
    while window.dims().iter().any(|&d| d != 1) {
        let num_dims = window.dims().len();
        for dim in 0..num_dims {
            if window.dims()[dim] == 1 {
                continue;
            }

            max_steps[dim] += 1;
            decomposition.push(dim);

            let (low, high) = window.split_into(dim);
            window = low;

            let lanes = high.lanes(0);
            let mut stream = SerializeStream::new();
            for lane in lanes {
                for elem in lane.as_slice().unwrap() {
                    elem.clone().serialize(&mut stream);
                }
            }

            let part_path = path.join(format!("block_part_{part_idx}.bin"));
            let part_file = std::fs::File::create(part_path).unwrap();
            stream.write_encode(compression, part_file).unwrap();

            part_idx += 1;
        }
    }

    let mut stream = SerializeStream::new();
    dims.serialize(&mut stream);
    max_steps.serialize(&mut stream);
    decomposition.serialize(&mut stream);

    let f = std::fs::File::create(path.join("block_header.bin")).unwrap();
    stream.write_encode(compression, f).unwrap();
}

pub(crate) fn load_block<T>(
    path: impl AsRef<Path>,
    steps: &[u32],
    low: T,
    filter: &(impl Filter<T> + Clone),
) -> (VolumeBlock<T>, Vec<usize>)
where
    T: Deserializable + Zero + Clone + Send,
{
    let path = path.as_ref();
    if !path.exists() || !path.is_dir() {
        panic!("Invalid path {path:?}");
    }

    let f = std::fs::File::open(path.join("block_header.bin")).unwrap();
    let stream = DeserializeStream::new_decode(f).unwrap();
    let mut stream = stream.stream();

    let dims = Vec::<usize>::deserialize(&mut stream);
    let max_steps = Vec::<u32>::deserialize(&mut stream);
    let decomposition = Vec::<usize>::deserialize(&mut stream);

    if steps.len() != dims.len() {
        panic!("Invalid number of steps {steps:?} for volume of size {dims:?}");
    }

    if steps.iter().zip(&max_steps).any(|(&s, &max)| s > max) {
        panic!("Can not load required number of steps {steps:?}, available {max_steps:?}");
    }

    let reconstructed_dims = steps.iter().map(|&s| 1 << s as usize).collect::<Vec<_>>();
    let mut block = VolumeBlock::new_fill(&reconstructed_dims, low).unwrap();

    if steps.iter().all(|&s| s == 0) {
        return (block, vec![]);
    }

    let mut new_decomp = vec![];
    let mut steps = Vec::from(steps);
    let mut skipped = vec![0; steps.len()];
    let mut block_window = block.window_mut();
    let mut block_part_dim = vec![1; steps.len()];
    let mut offsets = vec![1; steps.len()];
    for (i, dim) in decomposition.into_iter().enumerate().rev() {
        if steps[dim] == 0 {
            skipped[dim] += 1;
            block_part_dim[dim] *= 2;
            continue;
        }
        steps[dim] -= 1;

        let part_path = path.join(format!("block_part_{i}.bin"));
        let f = std::fs::File::open(part_path).unwrap();
        let stream = DeserializeStream::new_decode(f).unwrap();
        let mut stream = stream.stream();

        let elems = block_part_dim.iter().product::<usize>();
        let mut part = Vec::with_capacity(elems);
        for _ in 0..elems {
            part.push(T::deserialize(&mut stream));
        }

        let mut part = VolumeBlock::new_with_data(&block_part_dim, part).unwrap();
        let part_window = if skipped.iter().any(|&s| s != 0) {
            let cfg = WaveletDecompCfg::new(&skipped);
            let transform = WaveletTransform::new(filter.clone(), false);
            part = <WaveletTransform<_, _> as OneWayTransform<Forwards, _>>::apply(
                &transform, part, cfg,
            );

            let window = part.window();
            let window_range = window
                .dims()
                .iter()
                .zip(&skipped)
                .map(|(&d, &s)| d >> s)
                .collect::<Vec<_>>();

            window.into_custom_range(&window_range)
        } else {
            part.window()
        };

        let mut offset = vec![0; steps.len()];
        offset[dim] = offsets[dim];

        let mut coeff_window = block_window.custom_window_mut(&offset, part_window.dims());
        part_window.clone_to(&mut coeff_window);

        offsets[dim] *= 2;
        block_part_dim[dim] *= 2;
        new_decomp.insert(0, dim);
    }

    (block, new_decomp)
}
