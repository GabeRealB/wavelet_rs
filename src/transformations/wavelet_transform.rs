use num_traits::Zero;
use std::marker::PhantomData;

use super::{Backwards, Forwards, OneWayTransform};
use crate::{
    filter::{backwards_window, forwards_window, upscale_window, Filter},
    stream::{Deserializable, Serializable},
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

        assert!(dims.len() == cfg.backwards.len());
        for ((&dim, &f), &b) in dims.iter().zip(cfg.forwards).zip(cfg.backwards) {
            let required_size = 2usize.pow(b);
            assert!(b <= f);
            assert!(required_size <= dim);
        }

        if dims.iter().cloned().product::<usize>() == 0 {
            panic!("invalid number of steps");
        }

        if dims.iter().cloned().product::<usize>() == 1 || cfg.forwards.iter().all(|&f| f == 0) {
            return input;
        }

        let mut scratch = vec![T::zero(); *input.dims().iter().max().unwrap()];
        let (ops, output_size) =
            BackwardsOperation::new(cfg.forwards, cfg.backwards, cfg.start_dim, input.dims());
        let input_window = input.window_mut();

        self.back_(input_window, &mut scratch, &ops);

        let mut input_window = input.window_mut();
        for (dim, &size) in output_size.iter().enumerate() {
            let upscale_steps = (input_window.dims()[dim] / size).ilog2();
            for _ in 0..upscale_steps {
                upscale_window(dim, &mut input_window);
            }
        }

        input
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaveletRecompCfg<'a> {
    forwards: &'a [u32],
    backwards: &'a [u32],
    start_dim: usize,
}

impl<'a> WaveletRecompCfg<'a> {
    /// Constructs a new `WaveletRecompCfg`.
    pub fn new(forwards: &'a [u32], backwards: &'a [u32]) -> Self {
        Self::new_with_start_dim(forwards, backwards, 0)
    }

    /// Constructs a new `WaveletRecompCfg` with a custom starting direction.
    pub fn new_with_start_dim(forwards: &'a [u32], backwards: &'a [u32], start_dim: usize) -> Self {
        Self {
            forwards,
            backwards,
            start_dim,
        }
    }
}

impl<'a> From<WaveletDecompCfg<'a>> for WaveletRecompCfg<'a> {
    fn from(x: WaveletDecompCfg<'a>) -> Self {
        Self::new(x.steps, x.steps)
    }
}

impl<'a> From<&'a WaveletRecompCfgOwned> for WaveletRecompCfg<'a> {
    fn from(x: &'a WaveletRecompCfgOwned) -> Self {
        Self {
            forwards: &x.forwards,
            backwards: &x.backwards,
            start_dim: x.start_dim,
        }
    }
}

impl Serializable for WaveletRecompCfg<'_> {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.forwards.serialize(stream);
        self.backwards.serialize(stream)
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
    forwards: Vec<u32>,
    backwards: Vec<u32>,
    start_dim: usize,
}

impl WaveletRecompCfgOwned {
    /// Constructs a new `WaveletRecompCfgOwned`.
    pub fn new(forwards: Vec<u32>, backwards: Vec<u32>) -> Self {
        Self::new_with_start_dim(forwards, backwards, 0)
    }

    /// Constructs a new `WaveletRecompCfgOwned` with a custom starting direction.
    pub fn new_with_start_dim(forwards: Vec<u32>, backwards: Vec<u32>, start_dim: usize) -> Self {
        Self {
            forwards,
            backwards,
            start_dim,
        }
    }
}

impl From<WaveletRecompCfg<'_>> for WaveletRecompCfgOwned {
    fn from(x: WaveletRecompCfg<'_>) -> Self {
        Self::new(x.forwards.into(), x.backwards.into())
    }
}

impl Serializable for WaveletRecompCfgOwned {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.forwards.serialize(stream);
        self.backwards.serialize(stream);
        self.start_dim.serialize(stream);
    }
}

impl Deserializable for WaveletRecompCfgOwned {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let forwards = Deserializable::deserialize(stream);
        let backwards = Deserializable::deserialize(stream);
        let start_dim = Deserializable::deserialize(stream);
        Self {
            forwards,
            backwards,
            start_dim,
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
    fn new(
        forwards: &[u32],
        backwards: &[u32],
        start_dim: usize,
        dim: &[usize],
    ) -> (Vec<Self>, Vec<usize>) {
        assert_eq!(backwards.len(), forwards.len());

        let mut ops = Vec::new();
        let mut step = vec![0; backwards.len()];
        let mut countdown: Vec<_> = backwards
            .iter()
            .zip(forwards)
            .map(|(&b, &f)| f - b)
            .collect();

        let mut stop = false;
        while !stop {
            stop = true;

            for (i, (step, max)) in step[start_dim..]
                .iter_mut()
                .zip(&forwards[start_dim..])
                .enumerate()
            {
                if *step < *max {
                    *step += 1;
                    stop = false;

                    if countdown[start_dim + i] != 0 {
                        countdown[start_dim + i] -= 1;
                        ops.push(BackwardsOperation::Skip { dim: start_dim + i });
                    } else {
                        ops.push(BackwardsOperation::Backwards {
                            dim: start_dim + i,
                            adapt: None,
                        });
                    }
                }
            }

            for (i, (step, max)) in step[..start_dim]
                .iter_mut()
                .zip(&forwards[..start_dim])
                .enumerate()
            {
                if *step < *max {
                    *step += 1;
                    stop = false;

                    if countdown[i] != 0 {
                        countdown[i] -= 1;
                        ops.push(BackwardsOperation::Skip { dim: i });
                    } else {
                        ops.push(BackwardsOperation::Backwards {
                            dim: i,
                            adapt: None,
                        });
                    }
                }
            }
        }

        let base_size: Vec<_> = dim
            .iter()
            .zip(forwards)
            .map(|(&dim, &f)| dim >> f)
            .collect();

        let mut expected_size = base_size.clone();
        let mut curr_size = base_size;

        for op in ops.iter_mut().rev() {
            match op {
                BackwardsOperation::Backwards { dim, adapt } => {
                    if expected_size != curr_size {
                        let mut range = curr_size.clone();
                        range[*dim] *= 2;

                        let steps: Vec<_> = expected_size
                            .iter()
                            .zip(&curr_size)
                            .map(|(&ex, &curr)| (ex / curr).ilog2())
                            .collect();
                        *adapt = Some((range, steps));
                    }

                    expected_size[*dim] *= 2;
                    curr_size[*dim] *= 2;
                }
                BackwardsOperation::Skip { dim } => {
                    expected_size[*dim] *= 2;
                }
            }
        }

        (ops, curr_size)
    }

    fn dim(&self) -> usize {
        match self {
            BackwardsOperation::Backwards { dim, .. } => *dim,
            BackwardsOperation::Skip { dim } => *dim,
        }
    }
}
