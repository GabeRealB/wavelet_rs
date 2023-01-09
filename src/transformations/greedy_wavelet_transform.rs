use std::fs::File;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use std::path::Path;

use num_traits::{NumCast, Zero};

use super::wavelet_transform::ForwardsOperation;
use super::{
    Backwards, Forwards, OneWayTransform, ResampleCfg, ResampleIScale, ReversibleTransform,
};
use crate::filter::{AverageFilter, Filter};
use crate::range::for_each_range;
use crate::stream::{Deserializable, DeserializeStream, Serializable, SerializeStream};
use crate::volume::{Lane, LaneMut, VolumeBlock, VolumeWindow, VolumeWindowMut};

/// Definition of a filter to be used with the greedy wavelet transform.
pub trait GreedyFilter<Meta, T>: Sync {
    /// Applies a forward pass.
    fn forwards(
        &self,
        input: &Lane<'_, T>,
        meta_in: &Lane<'_, Meta>,
        low: &mut [T],
        high: &mut [T],
        meta: &mut [Meta],
    );

    /// Appliesa merge pass.
    fn merge(
        &self,
        input: &Lane<'_, T>,
        meta_in: &Lane<'_, Meta>,
        low: &mut [T],
        high: &mut [T],
        meta: &mut [Meta],
    );

    /// Applies a backwards pass, is the inverse of the forwards pass.
    fn backwards(&self, output: &mut LaneMut<'_, T>, low: &[T], high: &[T], meta_curr: &[Meta]);

    /// Initializes a suitable metadata block from the input data.
    fn compute_metadata(&self, input: VolumeWindow<'_, T>) -> VolumeBlock<Meta>;
}

/// Tries to construct a [`KnownGreedyFilter`] from an arbitrary type.
pub trait TryToKnownGreedyFilter {
    /// Fetches the [`KnownGreedyFilter`] which describes the desired operation.
    fn to_known_greedy_filter(&self) -> Option<KnownGreedyFilter>;
}

impl<T> TryToKnownGreedyFilter for T {
    default fn to_known_greedy_filter(&self) -> Option<KnownGreedyFilter> {
        None
    }
}

impl TryToKnownGreedyFilter for AverageFilter {
    fn to_known_greedy_filter(&self) -> Option<KnownGreedyFilter> {
        Some(KnownGreedyFilter::Average)
    }
}

impl TryToKnownGreedyFilter for KnownGreedyFilter {
    fn to_known_greedy_filter(&self) -> Option<KnownGreedyFilter> {
        Some(*self)
    }
}

/// Checks whether a type is associated with a [`KnownGreedyFilter`].
pub fn has_known_greedy_filter<T>(x: &T) -> bool {
    x.to_known_greedy_filter().is_some()
}

/// Known types of preimplemented greedy filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KnownGreedyFilter {
    /// Filter computing the average.
    Average,
}

impl<Meta, T> GreedyFilter<Meta, T> for KnownGreedyFilter
where
    AverageFilter: GreedyFilter<Meta, T>,
{
    fn forwards(
        &self,
        input: &Lane<'_, T>,
        meta_in: &Lane<'_, Meta>,
        low: &mut [T],
        high: &mut [T],
        meta: &mut [Meta],
    ) {
        match self {
            KnownGreedyFilter::Average => {
                GreedyFilter::forwards(&AverageFilter, input, meta_in, low, high, meta)
            }
        }
    }

    fn merge(
        &self,
        input: &Lane<'_, T>,
        meta_in: &Lane<'_, Meta>,
        low: &mut [T],
        high: &mut [T],
        meta: &mut [Meta],
    ) {
        match self {
            KnownGreedyFilter::Average => {
                GreedyFilter::merge(&AverageFilter, input, meta_in, low, high, meta)
            }
        }
    }

    fn backwards(&self, output: &mut LaneMut<'_, T>, low: &[T], high: &[T], meta_curr: &[Meta]) {
        match self {
            KnownGreedyFilter::Average => {
                GreedyFilter::backwards(&AverageFilter, output, low, high, meta_curr)
            }
        }
    }

    fn compute_metadata(&self, input: VolumeWindow<'_, T>) -> VolumeBlock<Meta> {
        match self {
            KnownGreedyFilter::Average => GreedyFilter::compute_metadata(&AverageFilter, input),
        }
    }
}

impl Serializable for KnownGreedyFilter {
    fn serialize(self, stream: &mut SerializeStream) {
        let num = self as i32;
        num.serialize(stream);
    }
}

impl Deserializable for KnownGreedyFilter {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        const AVERAGE_DISC: i32 = KnownGreedyFilter::Average as i32;

        let num = i32::deserialize(stream);
        match num {
            AVERAGE_DISC => Self::Average,
            _ => panic!(),
        }
    }
}

/// Metadata which keeps track of the number of elements contained in a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockCount(usize, usize);

impl BlockCount {
    /// Constructs a new block with the passed number
    /// of elements from the two sides.
    pub fn new(left: usize, right: usize) -> Self {
        Self(left, right)
    }

    /// Returns the number of elements contained.
    pub fn count(&self) -> usize {
        self.0 + self.1
    }

    /// Returns the number of elements originating from the left.
    pub fn left(&self) -> usize {
        self.0
    }

    /// Returns the number of elements originating from the right.
    pub fn right(&self) -> usize {
        self.1
    }
}

impl Add for BlockCount {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.left() + rhs.left(), self.right() + rhs.right())
    }
}

impl Zero for BlockCount {
    fn zero() -> Self {
        Self(0, 0)
    }

    fn is_zero(&self) -> bool {
        self.count().is_zero()
    }
}

impl Serializable for BlockCount {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.0.serialize(stream);
        self.1.serialize(stream);
    }
}

impl Deserializable for BlockCount {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let left = Deserializable::deserialize(stream);
        let right = Deserializable::deserialize(stream);

        Self(left, right)
    }
}

impl<T> GreedyFilter<BlockCount, T> for AverageFilter
where
    AverageFilter: Filter<T>,
    T: PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + NumCast
        + Clone,
{
    fn forwards(
        &self,
        input: &Lane<'_, T>,
        weight_in: &Lane<'_, BlockCount>,
        low: &mut [T],
        high: &mut [T],
        weight: &mut [BlockCount],
    ) {
        for (i, ((low, high), weight)) in low.iter_mut().zip(high).zip(weight).enumerate() {
            let idx_left = 2 * i;
            let idx_right = (2 * i) + 1;

            let left = input[idx_left].clone();
            let right = input[idx_right].clone();

            let left_count = weight_in[idx_left].count();
            let right_count = weight_in[idx_right].count();
            *weight = BlockCount::new(left_count, right_count);

            let left_weight = T::from(left_count).unwrap();
            let right_weight = T::from(right_count).unwrap();
            let new_weight = T::from(weight.count()).unwrap();

            let average = ((left_weight * left.clone()) + (right_weight * right)) / new_weight;
            let diff = left - average.clone();

            *low = average;
            *high = diff;
        }
    }

    fn merge(
        &self,
        input: &Lane<'_, T>,
        weight_in: &Lane<'_, BlockCount>,
        low: &mut [T],
        high: &mut [T],
        weight: &mut [BlockCount],
    ) {
        for (i, ((low, high), weight)) in low.iter_mut().zip(high).zip(weight).enumerate() {
            let idx_left = 2 * i;
            let idx_right = (2 * i) + 1;

            let left = input[idx_left].clone();
            let right = input[idx_right].clone();

            let left_count = weight_in[idx_left];
            let right_count = weight_in[idx_right];
            *weight = left_count + right_count;

            let left_weight = T::from(left_count.count()).unwrap();
            let right_weight = T::from(right_count.count()).unwrap();
            let new_weight = T::from(weight.count()).unwrap();

            let average = ((left_weight * left.clone()) + (right_weight * right)) / new_weight;
            let diff = left - average.clone();

            *low = average;
            *high = diff;
        }
    }

    fn backwards(&self, output: &mut LaneMut<'_, T>, low: &[T], high: &[T], meta: &[BlockCount]) {
        for (i, ((average, diff), &count)) in low.iter().zip(high).zip(meta).enumerate() {
            let idx_left = 2 * i;
            let idx_right = (2 * i) + 1;

            let left_weight = T::from(count.left()).unwrap();
            let right_weight = T::from(count.right()).unwrap();
            let weight = left_weight / right_weight;

            let left = average.clone() + diff.clone();
            let right = average.clone() - (weight * diff.clone());

            output[idx_left] = left;
            output[idx_right] = right;
        }
    }

    fn compute_metadata(&self, input: VolumeWindow<'_, T>) -> VolumeBlock<BlockCount> {
        VolumeBlock::new_fill(input.dims(), BlockCount::new(1, 0)).unwrap()
    }
}

/// Implementation of the greedy wavelet transform
/// which computes the wavelet transform for sizes other
/// than powers of two by decomposing the input into powers of two
/// and greedily recombining them when the combined size is a power
/// of two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GreedyWaveletTransform<F, Meta> {
    filter: F,
    _phantom: PhantomData<fn() -> Meta>,
}

/// Data needed to compute the inverse of the greedy wavelet transform.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GreedyWaveletTransformBackwardsCfg<'a> {
    steps: &'a [u32],
}

impl<'a> GreedyWaveletTransformBackwardsCfg<'a> {
    /// Constructs a new instance.
    pub fn new(steps: &'a [u32]) -> Self {
        Self { steps }
    }
}

impl<F, Meta> GreedyWaveletTransform<F, Meta> {
    /// Constructs a new instance which will use the provided filter.
    pub fn new(filter: F) -> Self {
        Self {
            filter,
            _phantom: PhantomData,
        }
    }
}

impl<Meta, T, F> OneWayTransform<Forwards, VolumeBlock<T>> for GreedyWaveletTransform<F, Meta>
where
    Meta: Zero + Send + Sync + Clone,
    T: Zero + Send + Sync + Clone,
    F: GreedyFilter<Meta, T>,
{
    type Result = GreedyTransformCoefficents<Meta, T>;

    type Cfg<'a> = ();

    fn apply(&self, input: VolumeBlock<T>, _: Self::Cfg<'_>) -> Self::Result {
        GreedyTransformCoefficents::new(input, &self.filter)
    }
}

impl<Meta, T, F> OneWayTransform<Backwards, &GreedyTransformCoefficents<Meta, T>>
    for GreedyWaveletTransform<F, Meta>
where
    Meta: Zero + Send + Sync + Clone,
    T: Zero + Send + Sync + Clone,
    F: GreedyFilter<Meta, T>,
{
    type Result = VolumeBlock<T>;

    type Cfg<'a> = GreedyWaveletTransformBackwardsCfg<'a>;

    fn apply(
        &self,
        input: &GreedyTransformCoefficents<Meta, T>,
        cfg: Self::Cfg<'_>,
    ) -> Self::Result {
        input.reconstruct(cfg.steps, &self.filter)
    }
}

impl<Meta, T, F> OneWayTransform<Backwards, GreedyTransformCoefficents<Meta, T>>
    for GreedyWaveletTransform<F, Meta>
where
    Meta: Zero + Send + Sync + Clone,
    T: Zero + Send + Sync + Clone,
    F: GreedyFilter<Meta, T>,
{
    type Result = VolumeBlock<T>;

    type Cfg<'a> = GreedyWaveletTransformBackwardsCfg<'a>;

    fn apply(
        &self,
        input: GreedyTransformCoefficents<Meta, T>,
        cfg: Self::Cfg<'_>,
    ) -> Self::Result {
        input.reconstruct(cfg.steps, &self.filter)
    }
}

/// Result of a variant of the wavelet transform which works
/// with sizes other than powers of two.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GreedyTransformCoefficents<Meta, T> {
    dims: Vec<usize>,
    pub(crate) low: VolumeBlock<T>,
    levels: Vec<CoeffLevel<Meta, T>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CoeffLevel<Meta, T> {
    dim: usize,
    dims: Vec<usize>,
    offsets: VolumeBlock<Vec<usize>>,
    metadata: VolumeBlock<VolumeBlock<Meta>>,
    coeff: VolumeBlock<Option<VolumeBlock<T>>>,
}

impl<Meta, T> GreedyTransformCoefficents<Meta, T>
where
    Meta: Zero + Send + Sync + Clone,
    T: Zero + Send + Sync + Clone,
{
    /// Applies the wavelet transform to the input block.
    pub fn new(input: VolumeBlock<T>, filter: &impl GreedyFilter<Meta, T>) -> Self {
        let steps = input
            .dims()
            .iter()
            .map(|&dim| dim.next_power_of_two().ilog2())
            .collect::<Vec<_>>();
        let metadata = filter.compute_metadata(input.window());
        Self::apply_forwards(input, metadata, filter, &steps, false)
    }

    fn apply_forwards(
        input: VolumeBlock<T>,
        mut metadata: VolumeBlock<Meta>,
        filter: &impl GreedyFilter<Meta, T>,
        steps: &[u32],
        adapt: bool,
    ) -> Self {
        let steps = ForwardsOperation::new(steps);

        let mut data = input;
        let dims = data.dims().to_vec();
        let mut coeff = Vec::new();

        for step in steps {
            let windows = power_of_two_windows(data.dims(), Some(step.dim));

            let data_window = data.window();
            let metadata_window = metadata.window();

            let (sx, rx) = std::sync::mpsc::sync_channel(windows.len());
            rayon::scope(|s| {
                for (i, window) in windows.iter().enumerate() {
                    let (window_size, window_offset, transformed_window_offset) = &window;

                    let data = data_window.custom_window(window_offset, window_size);
                    let meta = metadata_window.custom_window(window_offset, window_size);

                    let data = data.as_block();
                    let meta = meta.as_block();
                    let transformed_window_offset = transformed_window_offset.clone();
                    let sx = sx.clone();

                    // Block has reached minimum detail level along the dimension.
                    if window_size[step.dim] == 1 {
                        let _ = sx.send((i, transformed_window_offset, data, meta, None));
                        continue;
                    }

                    s.spawn(move |_| {
                        let data = data.window();
                        let meta = meta.window();
                        let (low, high, meta) = if adapt {
                            adjust_window(filter, data, meta, step.dim)
                        } else {
                            forwards_window(filter, data, meta, step.dim)
                        };
                        let _ = sx.send((i, transformed_window_offset, low, meta, Some(high)));
                    });
                }
            });

            let mut tmp = rx.try_iter().collect::<Vec<_>>();
            tmp.sort_by_key(|(i, _, _, _, _)| *i);

            let lvl_dims = Vec::from(data.dims());
            let mut lvl_offset = Vec::with_capacity(tmp.len());
            let mut lvl_data = Vec::with_capacity(tmp.len());
            let mut lvl_meta = Vec::with_capacity(tmp.len());
            let mut lvl_coeff = Vec::with_capacity(tmp.len());
            for (_, offset, data, meta, coeff) in tmp {
                lvl_offset.push(offset);
                lvl_data.push(data);
                lvl_meta.push(meta);
                lvl_coeff.push(coeff);
            }

            let num_windows = data_window
                .dims()
                .iter()
                .map(|&dim| num_power_of_two_decompositions(dim))
                .collect::<Vec<_>>();

            let lvl_offset = VolumeBlock::new_with_data(&num_windows, lvl_offset).unwrap();
            let lvl_data = VolumeBlock::new_with_data(&num_windows, lvl_data).unwrap();
            let lvl_meta = VolumeBlock::new_with_data(&num_windows, lvl_meta).unwrap();
            let lvl_coeff = VolumeBlock::new_with_data(&num_windows, lvl_coeff).unwrap();

            data = unroll_block(&lvl_data);
            metadata = unroll_block(&lvl_meta);

            coeff.push(CoeffLevel {
                dim: step.dim,
                dims: lvl_dims,
                offsets: lvl_offset,
                metadata: lvl_meta,
                coeff: lvl_coeff,
            });
        }

        Self {
            dims,
            low: data,
            levels: coeff,
        }
    }

    /// Applies the reverse transform and reconstructs the data to
    /// the desired detail level.
    pub fn reconstruct(
        &self,
        steps: &[u32],
        filter: &impl GreedyFilter<Meta, T>,
    ) -> VolumeBlock<T> {
        if steps.len() != self.dims.len()
            || steps
                .iter()
                .zip(&self.dims)
                .any(|(steps, dim)| *steps > dim.next_power_of_two().ilog2())
        {
            panic!(
                "Invalid number of steps {steps:?} for volume of size {:?}",
                self.dims
            );
        }

        let mut data = self.low.clone();
        let mut steps_left = Vec::from(steps);
        let mut steps_skipped = vec![0; steps.len()];
        for level in self.levels.iter().rev() {
            if steps_left[level.dim] == 0 {
                steps_skipped[level.dim] += 1;
                continue;
            }

            steps_left[level.dim] -= 1;

            let offsets = level.offsets.flatten();
            let metadata = level.metadata.flatten();
            let coeff = level.coeff.flatten();

            let data_window = data.window();
            let (sx, rx) = std::sync::mpsc::sync_channel(metadata.len());

            if steps_skipped.iter().all(|&skipped| skipped == 0) {
                rayon::scope(|s| {
                    for (i, ((offset, metadata), coeff)) in
                        offsets.iter().zip(metadata).zip(coeff).enumerate()
                    {
                        let window = data_window
                            .custom_window(offset, metadata.dims())
                            .as_block();
                        if let Some(coeff) = coeff {
                            let sx = sx.clone();
                            s.spawn(move |_| {
                                let window = window.window();
                                let metadata = metadata.window();
                                let coeff = coeff.window();
                                let window =
                                    backwards_window(filter, window, metadata, coeff, level.dim);
                                let _ = sx.send((i, window));
                            });
                        } else {
                            let _ = sx.send((i, window));
                        }
                    }
                });

                let mut windows = rx.try_iter().collect::<Vec<_>>();
                windows.sort_by_key(|(i, _)| *i);

                let data_vec = windows
                    .into_iter()
                    .map(|(_, window)| window)
                    .collect::<Vec<_>>();
                let data_tmp = VolumeBlock::new_with_data(level.offsets.dims(), data_vec).unwrap();
                data = unroll_block(&data_tmp);
            } else {
                let window_dims = level
                    .dims
                    .iter()
                    .enumerate()
                    .map(|(i, &dim)| {
                        if i == level.dim {
                            adjustment_block_dim(dim)
                        } else {
                            dim
                        }
                    })
                    .collect::<Vec<_>>();
                let coeff_block_dims = window_dims
                    .iter()
                    .map(|&dim| num_power_of_two_decompositions(dim))
                    .collect::<Vec<_>>();

                let metadata_window = level.metadata.window();
                let coeff_vec = coeff.iter().flatten().cloned().collect();

                let coeff_block = unroll_block(
                    &VolumeBlock::new_with_data(&coeff_block_dims, coeff_vec).unwrap(),
                );

                let meta_block = unroll_block(&metadata_window.as_block());
                let meta_block = meta_block.window();
                let meta_block = meta_block.custom_range(coeff_block.dims()).as_block();

                let adjusted =
                    Self::apply_forwards(coeff_block, meta_block, filter, &steps_skipped, true);

                let adjusted_window_dims = level
                    .dims
                    .iter()
                    .zip(&steps_skipped)
                    .map(|(&dim, &skipped)| adjust_block_size(dim, skipped))
                    .collect::<Vec<_>>();
                let data_dims = adjusted_window_dims
                    .iter()
                    .map(|&dim| num_power_of_two_decompositions(dim))
                    .collect::<Vec<_>>();

                let windows = power_of_two_windows(&adjusted_window_dims, Some(level.dim));

                let meta_block = unroll_block(&adjusted.levels.last().unwrap().metadata);
                let coeff_block = adjusted.low;

                let metadata = meta_block.window();
                let coeff = coeff_block.window();

                rayon::scope(|s| {
                    for (i, window) in windows.into_iter().enumerate() {
                        let (mut window_size, _, window_offset) = window;
                        let passthrough = window_size[level.dim] < 2;

                        if !passthrough {
                            let sx = sx.clone();
                            let metadata = metadata.clone();
                            let coeff = coeff.clone();
                            let data_window = data_window.clone();
                            s.spawn(move |_| {
                                window_size[level.dim] /= 2;
                                let window =
                                    data_window.custom_window(&window_offset, &window_size);
                                let metadata = metadata.custom_window(&window_offset, &window_size);
                                let coeff = coeff.custom_window(&window_offset, &window_size);
                                let window =
                                    backwards_window(filter, window, metadata, coeff, level.dim);
                                let _ = sx.send((i, window));
                            });
                        } else {
                            let window = data_window
                                .custom_window(&window_offset, &window_size)
                                .as_block();
                            let _ = sx.send((i, window));
                        }
                    }
                });

                let mut windows = rx.try_iter().collect::<Vec<_>>();
                windows.sort_by_key(|(i, _)| *i);

                let data_vec = windows
                    .into_iter()
                    .map(|(_, window)| window)
                    .collect::<Vec<_>>();
                let data_tmp = VolumeBlock::new_with_data(&data_dims, data_vec).unwrap();
                data = unroll_block(&data_tmp);
            }
        }

        data
    }

    /// Reconstructs the data to the desired detail level
    /// like [`reconstruct`], but resamples the output to
    /// have the same size as the original data.
    pub fn reconstruct_extend(
        &self,
        steps: &[u32],
        filter: &impl GreedyFilter<Meta, T>,
    ) -> VolumeBlock<T> {
        let mut data = self.reconstruct(steps, filter);
        if data.dims() == self.dims {
            return data;
        }

        let mut skipped_steps = steps
            .iter()
            .zip(&self.dims)
            .map(|(&st, &dim)| dim.next_power_of_two().ilog2() - st)
            .collect::<Vec<_>>();

        for i in 0..self.dims.len() {
            while skipped_steps[i] != 0 {
                skipped_steps[i] -= 1;

                let adjusted_window_dims = self
                    .dims
                    .iter()
                    .zip(&skipped_steps)
                    .map(|(&dim, &skipped)| adjust_block_size(dim, skipped))
                    .collect::<Vec<_>>();
                let data_dims = adjusted_window_dims
                    .iter()
                    .map(|&dim| num_power_of_two_decompositions(dim))
                    .collect::<Vec<_>>();

                let data_window = data.window();

                let windows = power_of_two_windows(&adjusted_window_dims, Some(i));
                let (sx, rx) = std::sync::mpsc::sync_channel(windows.len());

                rayon::scope(|s| {
                    for (j, window) in windows.into_iter().enumerate() {
                        let (mut window_size, _, window_offset) = window;
                        let passthrough = window_size[i] < 2;

                        if !passthrough {
                            let sx = sx.clone();
                            let data_window = data_window.clone();
                            s.spawn(move |_| {
                                let dst_size = window_size.clone();
                                let cfg = ResampleCfg::new(&dst_size);
                                window_size[i] /= 2;
                                let window =
                                    data_window.custom_window(&window_offset, &window_size);

                                let scaled = ResampleIScale.forwards(window.as_block(), cfg);
                                let _ = sx.send((j, scaled));
                            });
                        } else {
                            let window = data_window
                                .custom_window(&window_offset, &window_size)
                                .as_block();
                            let _ = sx.send((j, window));
                        }
                    }
                });

                let mut windows = rx.try_iter().collect::<Vec<_>>();
                windows.sort_by_key(|(i, _)| *i);

                let data_vec = windows
                    .into_iter()
                    .map(|(_, window)| window)
                    .collect::<Vec<_>>();
                let data_tmp = VolumeBlock::new_with_data(&data_dims, data_vec).unwrap();
                data = unroll_block(&data_tmp);
            }
        }

        data
    }
}

impl<Meta, T> GreedyTransformCoefficents<Meta, T>
where
    Meta: Serializable + Deserializable,
    T: Serializable + Deserializable,
{
    /// Writes out the coefficients to the filesystem.
    pub fn write_out(self, path: impl AsRef<Path>) {
        let path = path.as_ref();
        if !path.exists() || !path.is_dir() {
            panic!("Invalid path {path:?}");
        }

        let mut stream = SerializeStream::new();
        self.dims.serialize(&mut stream);
        self.low.serialize(&mut stream);

        self.levels
            .iter()
            .map(|lvl| lvl.dim)
            .collect::<Vec<_>>()
            .serialize(&mut stream);

        let header_path = path.join("block_header.bin");
        let header_file = File::create(header_path).unwrap();
        stream.write_encode(header_file).unwrap();

        for (i, level) in self.levels.into_iter().enumerate() {
            let mut stream = SerializeStream::new();
            level.serialize(&mut stream);

            let output_path = path.join(format!("block_part_{i}.bin"));
            let output_file = File::create(output_path).unwrap();
            stream.write_encode(output_file).unwrap();
        }
    }

    /// Reads in a subset of the data, reconstructible to
    /// the passed detail level, from the filesystem.
    pub fn read_for_steps(steps: &[u32], path: impl AsRef<Path>) -> Self
    where
        Meta: Default,
    {
        let path = path.as_ref();
        if !path.exists() || !path.is_dir() {
            panic!("Invalid path {path:?}");
        }

        let f = std::fs::File::open(path.join("block_header.bin")).unwrap();
        let stream = DeserializeStream::new_decode(f).unwrap();
        let mut stream = stream.stream();

        let dims = Vec::<usize>::deserialize(&mut stream);
        let low = Deserializable::deserialize(&mut stream);
        let level_dims = Vec::<usize>::deserialize(&mut stream);

        if steps.len() != dims.len()
            || steps
                .iter()
                .zip(&dims)
                .any(|(steps, dim)| *steps > dim.next_power_of_two().ilog2())
        {
            panic!("Invalid number of steps {steps:?} for volume of size {dims:?}");
        }

        let num_levels = dims
            .iter()
            .map(|d| d.next_power_of_two().ilog2())
            .fold(0, |acc, x| acc + (x as usize));

        let mut levels = (0..num_levels)
            .zip(level_dims)
            .map(|(_, dim)| CoeffLevel {
                dim,
                dims: vec![],
                offsets: VolumeBlock::new_default(&[1]).unwrap(),
                metadata: VolumeBlock::new_with_data(
                    &[1],
                    vec![VolumeBlock::new_with_data(&[1], vec![Default::default()]).unwrap()],
                )
                .unwrap(),
                coeff: VolumeBlock::new_with_data(&[1], vec![None]).unwrap(),
            })
            .collect::<Vec<_>>();

        let mut steps_left = Vec::from(steps);
        for (i, level) in levels.iter_mut().enumerate().rev() {
            if steps_left[level.dim] == 0 {
                continue;
            } else {
                steps_left[level.dim] -= 1;
            }

            let f = std::fs::File::open(path.join(format!("block_part_{i}.bin"))).unwrap();
            let stream = DeserializeStream::new_decode(f).unwrap();
            let mut stream = stream.stream();

            *level = Deserializable::deserialize(&mut stream);
        }

        Self { dims, low, levels }
    }
}

impl<Meta, T> Serializable for GreedyTransformCoefficents<Meta, T>
where
    Meta: Serializable,
    T: Serializable,
{
    fn serialize(self, stream: &mut SerializeStream) {
        self.dims.serialize(stream);
        self.low.serialize(stream);
        self.levels.serialize(stream);
    }
}

impl<Meta, T> Deserializable for GreedyTransformCoefficents<Meta, T>
where
    Meta: Deserializable,
    T: Deserializable,
{
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let dims = Deserializable::deserialize(stream);
        let low = Deserializable::deserialize(stream);
        let levels = Deserializable::deserialize(stream);

        Self { dims, low, levels }
    }
}

impl<Meta, T> Serializable for CoeffLevel<Meta, T>
where
    Meta: Serializable,
    T: Serializable,
{
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.dim.serialize(stream);
        self.dims.serialize(stream);
        self.offsets.serialize(stream);
        self.metadata.serialize(stream);
        self.coeff.serialize(stream);
    }
}

impl<Meta, T> Deserializable for CoeffLevel<Meta, T>
where
    Meta: Deserializable,
    T: Deserializable,
{
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let dim = Deserializable::deserialize(stream);
        let dims = Deserializable::deserialize(stream);
        let offsets = Deserializable::deserialize(stream);
        let metadata = Deserializable::deserialize(stream);
        let coeff = Deserializable::deserialize(stream);

        Self {
            dim,
            dims,
            offsets,
            metadata,
            coeff,
        }
    }
}

fn forwards_window<Meta, T>(
    f: &impl GreedyFilter<Meta, T>,
    data: VolumeWindow<'_, T>,
    meta: VolumeWindow<'_, Meta>,
    dim: usize,
) -> (VolumeBlock<T>, VolumeBlock<T>, VolumeBlock<Meta>)
where
    T: Zero + Clone,
    Meta: Zero + Clone,
{
    let arr_len = data.dims()[dim] / 2;
    let mut data_scratch = vec![T::zero(); arr_len * 2];
    let (data_low, data_high) = data_scratch.split_at_mut(arr_len);

    let mut meta_scratch = vec![Meta::zero(); arr_len];

    let data_lanes = data.lanes(dim);
    let meta_lanes = meta.lanes(dim);

    let mut output_dims = Vec::from(data.dims());
    output_dims[dim] = arr_len;

    let mut low = VolumeBlock::new_zero(&output_dims).unwrap();
    let mut high = VolumeBlock::new_zero(&output_dims).unwrap();
    let mut meta = VolumeBlock::new_zero(&output_dims).unwrap();

    let mut low_window = low.window_mut();
    let mut high_window = high.window_mut();
    let mut meta_window = meta.window_mut();

    let low_lanes = low_window.lanes_mut(dim);
    let high_lanes = high_window.lanes_mut(dim);
    let meta_lanes_out = meta_window.lanes_mut(dim);

    for ((((data, meta), mut low), mut high), mut meta_out) in data_lanes
        .zip(meta_lanes)
        .zip(low_lanes)
        .zip(high_lanes)
        .zip(meta_lanes_out)
    {
        f.forwards(&data, &meta, data_low, data_high, &mut meta_scratch);

        for (src, dst) in data_low.iter().zip(low.into_iter()) {
            *dst = src.clone();
        }
        for (src, dst) in data_high.iter().zip(high.into_iter()) {
            *dst = src.clone();
        }
        for (src, dst) in meta_scratch.iter().zip(meta_out.into_iter()) {
            *dst = src.clone();
        }
    }

    (low, high, meta)
}

fn adjust_window<Meta, T>(
    f: &impl GreedyFilter<Meta, T>,
    data: VolumeWindow<'_, T>,
    meta: VolumeWindow<'_, Meta>,
    dim: usize,
) -> (VolumeBlock<T>, VolumeBlock<T>, VolumeBlock<Meta>)
where
    T: Zero + Clone,
    Meta: Zero + Clone,
{
    let arr_len = data.dims()[dim] / 2;
    let mut data_scratch = vec![T::zero(); arr_len * 2];
    let (data_low, data_high) = data_scratch.split_at_mut(arr_len);

    let mut meta_scratch = vec![Meta::zero(); arr_len];

    let data_lanes = data.lanes(dim);
    let meta_lanes = meta.lanes(dim);

    let mut output_dims = Vec::from(data.dims());
    output_dims[dim] = arr_len;

    let mut low = VolumeBlock::new_zero(&output_dims).unwrap();
    let mut high = VolumeBlock::new_zero(&output_dims).unwrap();
    let mut meta = VolumeBlock::new_zero(&output_dims).unwrap();

    let mut low_window = low.window_mut();
    let mut high_window = high.window_mut();
    let mut meta_window = meta.window_mut();

    let low_lanes = low_window.lanes_mut(dim);
    let high_lanes = high_window.lanes_mut(dim);
    let meta_lanes_out = meta_window.lanes_mut(dim);

    for ((((data, meta), mut low), mut high), mut meta_out) in data_lanes
        .zip(meta_lanes)
        .zip(low_lanes)
        .zip(high_lanes)
        .zip(meta_lanes_out)
    {
        f.merge(&data, &meta, data_low, data_high, &mut meta_scratch);

        for (src, dst) in data_low.iter().zip(low.into_iter()) {
            *dst = src.clone();
        }
        for (src, dst) in data_high.iter().zip(high.into_iter()) {
            *dst = src.clone();
        }
        for (src, dst) in meta_scratch.iter().zip(meta_out.into_iter()) {
            *dst = src.clone();
        }
    }

    (low, high, meta)
}

fn backwards_window<Meta, T>(
    f: &impl GreedyFilter<Meta, T>,
    data: VolumeWindow<'_, T>,
    meta: VolumeWindow<'_, Meta>,
    coeff: VolumeWindow<'_, T>,
    dim: usize,
) -> VolumeBlock<T>
where
    T: Zero + Clone,
    Meta: Zero + Clone,
{
    let scratch_size = data.dims()[dim];
    let mut data_scratch = vec![T::zero(); scratch_size * 2];
    let mut meta_scratch = vec![Meta::zero(); scratch_size];

    let (data_low, data_high) = data_scratch.split_at_mut(scratch_size);

    let mut output_size = Vec::from(data.dims());
    output_size[dim] *= 2;

    let mut output = VolumeBlock::new_zero(&output_size).unwrap();
    let output_window = output.window_mut();

    let apply = |data: VolumeWindow<'_, T>,
                 coeff: VolumeWindow<'_, T>,
                 meta: VolumeWindow<'_, Meta>,
                 low: &mut [T],
                 high: &mut [T],
                 meta_scratch: &mut [Meta],
                 mut output: VolumeWindowMut<'_, T>| {
        let data_lanes = data.lanes(dim);
        let coeff_lanes = coeff.lanes(dim);
        let meta_lanes = meta.lanes(dim);

        let output_lanes = output.lanes_mut(dim);

        for (((data, coeff), meta), mut output) in data_lanes
            .zip(coeff_lanes)
            .zip(meta_lanes)
            .zip(output_lanes)
        {
            for (src, dst) in data.iter().zip(low.iter_mut()) {
                *dst = src.clone();
            }
            for (src, dst) in coeff.iter().zip(high.iter_mut()) {
                *dst = src.clone();
            }
            for (src, dst) in meta.iter().zip(meta_scratch.iter_mut()) {
                *dst = src.clone();
            }

            f.backwards(&mut output, low, high, meta_scratch);
        }
    };

    // if data.dims() != coeff.dims() {
    //     let mut meta = meta.as_block();
    //     let mut coeff = coeff.as_block();

    //     let skipped = data
    //         .dims()
    //         .iter()
    //         .zip(coeff.dims())
    //         .map(|(&data, &coeff)| (coeff / data).ilog2())
    //         .collect::<Vec<_>>();
    //     for (dim, skipped) in skipped.into_iter().enumerate() {
    //         for _ in 0..skipped {
    //             let (low, _, meta_) = adjust_window(f, coeff.window(), meta.window(), dim);
    //             meta = meta_;
    //             coeff = low;
    //         }
    //     }

    //     apply(
    //         data,
    //         coeff.window(),
    //         meta.window(),
    //         data_low,
    //         data_high,
    //         &mut meta_scratch,
    //         output_window,
    //     );
    // } else {
    //     apply(
    //         data,
    //         coeff,
    //         meta,
    //         data_low,
    //         data_high,
    //         &mut meta_scratch,
    //         output_window,
    //     );
    // }

    apply(
        data,
        coeff,
        meta,
        data_low,
        data_high,
        &mut meta_scratch,
        output_window,
    );

    output
}

fn unroll_block<T>(block: &VolumeBlock<VolumeBlock<T>>) -> VolumeBlock<T>
where
    T: Zero + Clone,
{
    let range = block.dims().iter().map(|&d| 0..d).collect::<Vec<_>>();
    let mut offsets = VolumeBlock::<Vec<usize>>::new_default(block.dims()).unwrap();

    let last_idx = |idx: &[usize]| {
        let mut idx = Vec::from(idx);

        for (i, dim_idx) in idx.iter_mut().enumerate().rev() {
            if *dim_idx > 0 {
                *dim_idx -= 1;
                return (idx, i);
            }
        }

        (idx, 0)
    };

    for_each_range(range.iter().cloned(), |idx| {
        if idx.iter().all(|&i| i == 0) {
            offsets[idx] = vec![0; offsets.dims().len()];
            return;
        }

        let (last_idx, dim) = last_idx(idx);
        let mut offset = offsets[&*last_idx].clone();
        offset[dim] += block[&*last_idx].dims()[dim];

        offsets[idx] = offset;
    });

    let last_offset = offsets.flatten().last().unwrap();
    let last_block_size = block.flatten().last().unwrap().dims();

    let dims = last_offset
        .iter()
        .zip(last_block_size)
        .map(|(off, size)| off + size)
        .collect::<Vec<_>>();

    let mut data = VolumeBlock::<T>::new_zero(&dims).unwrap();
    let mut data_window = data.window_mut();

    for_each_range(range.into_iter(), |idx| {
        let block = &block[idx];
        let block_window = block.window();

        let block_offset = &offsets[idx];
        let block_size = block_window.dims();

        let mut window = data_window.custom_window_mut(block_offset, block_size);
        block_window.clone_to(&mut window);
    });

    data
}

fn num_power_of_two_decompositions(num: usize) -> usize {
    let mut rest = num;
    let mut count = 0;

    while rest > 0 {
        let last_pow2: usize =
            1 << ((8 * std::mem::size_of::<usize>() - 1) - rest.leading_zeros() as usize);

        rest -= last_pow2;
        count += 1;
    }

    count
}

fn adjustment_block_dim(dim: usize) -> usize {
    let mut rest = dim & !1;
    let mut dim = 0;

    while rest > 0 {
        let last_pow2: usize =
            1 << ((8 * std::mem::size_of::<usize>() - 1) - rest.leading_zeros() as usize);

        rest -= last_pow2;
        dim += last_pow2 / 2;
    }

    dim
}

fn adjust_block_size(mut size: usize, skipped: u32) -> usize {
    for _ in 0..skipped {
        let mut rest = size;
        let mut new_size = 0;

        while rest > 0 {
            let last_pow2: usize =
                1 << ((8 * std::mem::size_of::<usize>() - 1) - rest.leading_zeros() as usize);

            rest -= last_pow2;

            if last_pow2 > 1 {
                new_size += last_pow2 / 2;
            } else {
                new_size += last_pow2;
            }
        }

        size = new_size;
    }

    size
}

fn power_of_two_decompositions_and_offsets(
    num: usize,
    transform: bool,
) -> Vec<(usize, usize, usize)> {
    let mut rest = num;
    let mut offset = 0;
    let mut transformed_offset = 0;
    let mut decomps = Vec::new();

    while rest > 0 {
        let last_pow2: usize =
            1 << ((8 * std::mem::size_of::<usize>() - 1) - rest.leading_zeros() as usize);

        decomps.push((last_pow2, offset, transformed_offset));
        rest -= last_pow2;
        offset += last_pow2;

        if last_pow2 > 1 && transform {
            transformed_offset += last_pow2 / 2;
        } else {
            transformed_offset += last_pow2;
        }
    }

    decomps
}

fn power_of_two_windows(
    dims: &[usize],
    dim: Option<usize>,
) -> Vec<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let decomps = dims
        .iter()
        .enumerate()
        .map(|(i, &d)| power_of_two_decompositions_and_offsets(d, i == dim.unwrap_or(!i)))
        .collect::<Vec<_>>();
    let ranges = decomps.iter().map(|dec| 0..dec.len()).collect::<Vec<_>>();

    let mut windows = Vec::new();
    for_each_range(ranges.into_iter(), |idx| {
        let size = idx.iter().zip(&decomps).map(|(&i, dec)| dec[i].0).collect();
        let offset = idx.iter().zip(&decomps).map(|(&i, dec)| dec[i].1).collect();
        let transformed_offset = idx.iter().zip(&decomps).map(|(&i, dec)| dec[i].2).collect();
        windows.push((size, offset, transformed_offset))
    });

    windows
}

#[cfg(test)]
mod tests {
    use crate::{filter::AverageFilter, volume::VolumeBlock};

    use super::GreedyTransformCoefficents;

    #[test]
    fn test_greedy() {
        let data = VolumeBlock::<f32>::new_with_data(
            &[7, 7],
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0,
                3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0,
                5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
                7.0,
            ],
        )
        .unwrap();

        let coeff = GreedyTransformCoefficents::new(data, &AverageFilter);
        let _data = coeff.reconstruct(&[3, 2], &AverageFilter);
        let _data = coeff.reconstruct_extend(&[3, 2], &AverageFilter);
    }
}
