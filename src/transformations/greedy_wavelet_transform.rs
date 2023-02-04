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
use crate::stream::{
    CompressionLevel, Deserializable, DeserializeStream, Serializable, SerializeStream,
};
use crate::volume::{Lane, LaneMut, VolumeBlock, VolumeWindow};

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

/// A [`GreedyFilter`] where the metadata can be derived from only
/// the data size and the metadata of the previous step.
pub trait DerivableMetadataFilter<Meta, T>: GreedyFilter<Meta, T> {
    /// Derives the initial metadata block.
    fn derive_init(&self, dims: &[usize]) -> VolumeBlock<Meta>;

    /// Derives the metadata of the next step.
    fn derive_step(&self, input: Lane<'_, Meta>, output: LaneMut<'_, Meta>);
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

impl<Meta, T> DerivableMetadataFilter<Meta, T> for KnownGreedyFilter
where
    KnownGreedyFilter: GreedyFilter<Meta, T>,
    AverageFilter: DerivableMetadataFilter<Meta, T>,
{
    fn derive_init(&self, dims: &[usize]) -> VolumeBlock<Meta> {
        match self {
            KnownGreedyFilter::Average => {
                DerivableMetadataFilter::derive_init(&AverageFilter, dims)
            }
        }
    }

    fn derive_step(&self, input: Lane<'_, Meta>, output: LaneMut<'_, Meta>) {
        match self {
            KnownGreedyFilter::Average => {
                DerivableMetadataFilter::derive_step(&AverageFilter, input, output)
            }
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
        self.derive_init(input.dims())
    }
}

impl<T> DerivableMetadataFilter<BlockCount, T> for AverageFilter
where
    AverageFilter: GreedyFilter<BlockCount, T>,
{
    fn derive_init(&self, dims: &[usize]) -> VolumeBlock<BlockCount> {
        VolumeBlock::new_fill(dims, BlockCount::new(1, 0)).unwrap()
    }

    fn derive_step(&self, input: Lane<'_, BlockCount>, mut output: LaneMut<'_, BlockCount>) {
        for (i, out) in output.into_iter().enumerate() {
            let left = 2 * i;
            let right = (2 * i) + 1;
            *out = BlockCount::new(input[left].count(), input[right].count());
        }
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
    pub(crate) meta: VolumeBlock<Meta>,
    levels: Vec<CoeffLevel<Meta, T>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CoeffLevel<Meta, T> {
    dim: usize,
    dims: Vec<usize>,
    split: Option<usize>,
    coeff: VolumeBlock<T>,
    metadata: VolumeBlock<Meta>,
}

impl<Meta, T> GreedyTransformCoefficents<Meta, T>
where
    Meta: Zero + Clone,
    T: Zero + Clone,
{
    /// Applies the wavelet transform to the input block.
    pub fn new(input: VolumeBlock<T>, filter: &impl GreedyFilter<Meta, T>) -> Self {
        let metadata = filter.compute_metadata(input.window());
        Self::new_with_meta(input, metadata, filter)
    }

    pub(crate) fn new_with_meta(
        input: VolumeBlock<T>,
        metadata: VolumeBlock<Meta>,
        filter: &impl GreedyFilter<Meta, T>,
    ) -> Self {
        let steps = input
            .dims()
            .iter()
            .map(|&dim| dim.next_power_of_two().ilog2())
            .collect::<Vec<_>>();
        Self::apply_forwards(input, metadata, filter, &steps, false)
    }

    pub(crate) fn merge(
        self,
        mut other: Self,
        dim: usize,
        filter: &impl GreedyFilter<Meta, T>,
    ) -> Self {
        assert!(self
            .dims
            .iter()
            .zip(&other.dims)
            .enumerate()
            .all(|(i, (&l, &r))| i == dim || l == r));
        assert!(self.levels.len() >= other.levels.len());

        let mut base_dims = Vec::from(self.low.dims());
        base_dims[dim] += other.low.dims()[dim];

        fn merge_block<T: Clone + Zero>(
            dim: usize,
            dims: &[usize],
            left: VolumeBlock<T>,
            right: VolumeBlock<T>,
        ) -> VolumeBlock<T> {
            let mut merged = VolumeBlock::new_zero(dims).unwrap();
            let mut window = merged.window_mut();

            let left_window = left.window();
            let right_window = right.window();

            let mut part = window.custom_window_mut(&vec![0; dims.len()], left_window.dims());
            left_window.clone_to(&mut part);

            let mut offset = vec![0; dims.len()];
            offset[dim] = left_window.dims()[dim];

            let mut part = window.custom_window_mut(&offset, right_window.dims());
            right_window.clone_to(&mut part);

            merged
        }

        let low = merge_block(dim, &base_dims, self.low, other.low);
        let meta = merge_block(dim, &base_dims, self.meta, other.meta);

        let dims = self.dims;
        let mut steps = vec![0; dims.len()];
        steps[dim] = 1;

        let adapted = Self::apply_forwards(low, meta, filter, &steps, true);
        let low = adapted.low;
        let meta = adapted.meta;

        let mut levels = Vec::new();
        for l_lvl in self.levels {
            let r_lvl_pos = other.levels.iter().position(|lvl| lvl.dim == l_lvl.dim);

            if let Some(r_lvl_pos) = r_lvl_pos {
                let r_lvl = other.levels.remove(r_lvl_pos);

                assert!(l_lvl
                    .dims
                    .iter()
                    .zip(&r_lvl.dims)
                    .enumerate()
                    .all(|(i, (&l, &r))| i == dim || l == r));

                let mut metadata_dims = Vec::from(l_lvl.metadata.dims());
                metadata_dims[dim] += r_lvl.metadata.dims()[dim];

                let mut coeff_dims = Vec::from(l_lvl.coeff.dims());
                coeff_dims[dim] += r_lvl.coeff.dims()[dim];

                let metadata = if metadata_dims
                    .iter()
                    .zip(l_lvl.metadata.dims())
                    .zip(r_lvl.metadata.dims())
                    .enumerate()
                    .all(|(i, ((&d, &l), &r))| {
                        (i == dim && d == l + r) || (i != dim && d == l && d == r)
                    }) {
                    merge_block(dim, &metadata_dims, l_lvl.metadata, r_lvl.metadata)
                } else {
                    let meta_window = r_lvl.metadata.window();
                    let r_meta = meta_window.custom_range(l_lvl.metadata.dims()).as_block();
                    merge_block(dim, &metadata_dims, l_lvl.metadata, r_meta)
                };

                let coeff = merge_block(dim, &coeff_dims, l_lvl.coeff, r_lvl.coeff);

                let metadata_window = metadata.window();
                let metadata_part = metadata_window.custom_range(&coeff_dims).as_block();
                let adapted = Self::apply_forwards(coeff, metadata_part, filter, &steps, true);

                let metadata_part = adapted.meta;
                let coeff = adapted.low;

                let metadata = if coeff_dims == metadata_dims {
                    metadata_part
                } else {
                    let mut meta = VolumeBlock::new_zero(&l_lvl.dims).unwrap();
                    let mut meta_window = meta.window_mut();
                    let metadata_part_window = metadata_part.window();

                    metadata_part_window
                        .clone_to(&mut meta_window.custom_range_mut(metadata_part_window.dims()));

                    let mut offset = vec![0; metadata_window.dims().len()];
                    offset[dim] = metadata_window.dims()[dim] - 1;

                    let mut range = Vec::from(meta_window.dims());
                    range[dim] = 1;

                    let metadata_window = metadata_window.custom_window(&offset, &range);

                    offset[dim] = (metadata_window.dims()[dim] - 1) / 2;
                    metadata_window.clone_to(&mut meta_window.custom_window_mut(&offset, &range));

                    meta
                };

                let dim = l_lvl.dim;
                let dims = l_lvl.dims;
                let split = l_lvl.split;

                levels.push(CoeffLevel {
                    dim,
                    dims,
                    split,
                    coeff,
                    metadata,
                });
            } else {
                levels.push(l_lvl);
            }
        }

        Self {
            dims,
            low,
            meta,
            levels,
        }
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
            let data_window = data.window();
            let metadata_window = metadata.window();

            let mut aggregated_dims = Vec::from(data.dims());
            aggregated_dims[step.dim] =
                (aggregated_dims[step.dim] / 2) + (aggregated_dims[step.dim] % 2);
            let aggregated_dims = aggregated_dims;

            let (split, transform_range) = if data.dims()[step.dim] % 2 == 0 {
                (None, Vec::from(data.dims()))
            } else {
                let mut range = Vec::from(data.dims());
                range[step.dim] &= !1;

                (Some(data.dims()[step.dim] & !1), range)
            };

            let transform_data = data_window.custom_range(&transform_range);
            let transform_meta = metadata_window.custom_range(&transform_range);
            let (low, high, mut meta) = if adapt {
                adjust_window(filter, transform_data, transform_meta, step.dim)
            } else {
                forwards_window(filter, transform_data, transform_meta, step.dim)
            };

            let mut aggreagated_data = VolumeBlock::new_zero(&aggregated_dims).unwrap();
            let mut aggreagated_meta = VolumeBlock::new_zero(&aggregated_dims).unwrap();

            let mut aggreagated_data_window = aggreagated_data.window_mut();
            let mut aggreagated_meta_window = aggreagated_meta.window_mut();

            let low_window = low.window();
            let meta_window = meta.window();

            low_window.clone_to(&mut aggreagated_data_window.custom_range_mut(low_window.dims()));
            meta_window.clone_to(&mut aggreagated_meta_window.custom_range_mut(meta_window.dims()));

            if let Some(split) = split {
                let mut offset = vec![0; transform_range.len()];
                offset[step.dim] = split;

                let mut range = Vec::from(data.dims());
                range[step.dim] = 1;

                let data_window_part = data_window.custom_window(&offset, &range);
                let metadata_window_part = metadata_window.custom_window(&offset, &range);

                offset[step.dim] = split / 2;
                data_window_part
                    .clone_to(&mut aggreagated_data_window.custom_window_mut(&offset, &range));
                metadata_window_part
                    .clone_to(&mut aggreagated_meta_window.custom_window_mut(&offset, &range));

                meta = aggreagated_meta_window.as_block();
            }

            coeff.push(CoeffLevel {
                dim: step.dim,
                dims: data.dims().into(),
                split,
                coeff: high,
                metadata: meta,
            });

            data = aggreagated_data;
            metadata = aggreagated_meta;
        }

        Self {
            dims,
            low: data,
            meta: metadata,
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
        self.reconstruct_with_low(self.low.clone(), steps, filter)
    }

    pub(crate) fn reconstruct_with_low(
        &self,
        low: VolumeBlock<T>,
        steps: &[u32],
        filter: &impl GreedyFilter<Meta, T>,
    ) -> VolumeBlock<T> {
        assert_eq!(low.dims(), self.low.dims());

        if steps.len() != self.dims.len() {
            panic!(
                "Invalid number of steps {steps:?} for volume of size {:?}",
                self.dims
            );
        }

        let mut data = low;
        let mut steps_left = Vec::from(steps);
        let mut steps_skipped = vec![0; steps.len()];

        for level in self.levels.iter().rev() {
            if steps_left[level.dim] == 0 {
                steps_skipped[level.dim] += 1;
                continue;
            }

            steps_left[level.dim] -= 1;

            let data_window = data.window();
            let metadata_window = level.metadata.window();
            let reconstructed = if steps_skipped.iter().all(|&skipped| skipped == 0) {
                let data_window_part = data_window.custom_range(level.coeff.dims());
                let metadata_window = metadata_window.custom_range(level.coeff.dims());
                let coeff_window = level.coeff.window();

                backwards_window(
                    filter,
                    data_window_part,
                    metadata_window,
                    coeff_window,
                    level.dim,
                )
            } else {
                let coeff = level.coeff.clone();
                let metadata = level.metadata.window();
                let metadata = metadata.custom_range(coeff.dims()).as_block();

                let aggreagated =
                    Self::apply_forwards(coeff, metadata, filter, &steps_skipped, true);

                let coeff = aggreagated.low;
                let metadata = aggreagated.meta;

                let data_window_part = data_window.custom_range(coeff.dims());
                let metadata_window = metadata.window();
                let coeff_window = coeff.window();

                backwards_window(
                    filter,
                    data_window_part,
                    metadata_window,
                    coeff_window,
                    level.dim,
                )
            };

            if let Some(split) = level.split {
                let reconstructed_window = reconstructed.window();

                let mut reconstructed_dims = Vec::from(reconstructed_window.dims());
                reconstructed_dims[level.dim] = level.dims[level.dim];

                let mut reconstructed_data = VolumeBlock::new_zero(&reconstructed_dims).unwrap();
                let mut reconstructed_data_window = reconstructed_data.window_mut();

                reconstructed_window.clone_to(
                    &mut reconstructed_data_window.custom_range_mut(reconstructed_window.dims()),
                );

                let mut offset = vec![0; reconstructed_data_window.dims().len()];
                offset[level.dim] = split / 2;

                let mut range = Vec::from(reconstructed_data_window.dims());
                range[level.dim] = 1;

                let data_window = data_window.custom_window(&offset, &range);

                offset[level.dim] = split;
                data_window
                    .clone_to(&mut reconstructed_data_window.custom_window_mut(&offset, &range));

                data = reconstructed_data;
            } else {
                data = reconstructed;
            }
        }

        data
    }

    /// Reconstructs the data to the desired detail level
    /// like [`reconstruct`](Self::reconstruct), but resamples the output to
    /// have the same size as the original data.
    pub fn reconstruct_extend(
        &self,
        steps: &[u32],
        filter: &impl GreedyFilter<Meta, T>,
    ) -> VolumeBlock<T> {
        self.reconstruct_extend_to(steps, &self.dims, filter)
    }

    pub(crate) fn reconstruct_extend_to(
        &self,
        steps: &[u32],
        dims: &[usize],
        filter: &impl GreedyFilter<Meta, T>,
    ) -> VolumeBlock<T> {
        self.reconstruct_extend_to_with_low(self.low.clone(), steps, dims, filter)
    }

    pub(crate) fn reconstruct_extend_to_with_low(
        &self,
        low: VolumeBlock<T>,
        steps: &[u32],
        dims: &[usize],
        filter: &impl GreedyFilter<Meta, T>,
    ) -> VolumeBlock<T> {
        let mut data = self.reconstruct_with_low(low, steps, filter);
        if data.dims() == dims {
            return data;
        }

        let mut skipped_steps = data
            .dims()
            .iter()
            .zip(dims)
            .map(|(&data_dim, &dim)| {
                let st = data_dim.next_power_of_two().ilog2();
                let dim_steps = dim.next_power_of_two().ilog2();
                if dim_steps <= st {
                    0
                } else {
                    dim_steps - st
                }
            })
            .collect::<Vec<_>>();

        for i in 0..dims.len() {
            while skipped_steps[i] != 0 {
                skipped_steps[i] -= 1;

                let mut extended_dims = Vec::from(data.dims());
                extended_dims[i] = adjust_block_size(dims[i], skipped_steps[i]);

                let data_window = data.window();
                if extended_dims[i] % 2 == 0 {
                    let extended_range = extended_dims[i];
                    extended_dims[i] = extended_range.next_multiple_of(data_window.dims()[i]);

                    let extended = ResampleIScale
                        .forwards(data_window.as_block(), ResampleCfg::new(&extended_dims));
                    let extended_window = extended.window();

                    extended_dims[i] = extended_range;
                    data = extended_window.into_custom_range(&extended_dims).as_block();

                    /* data = ResampleIScale
                    .forwards(data_window.as_block(), ResampleCfg::new(&extended_dims)); */
                } else {
                    let mut extend_to_dims = extended_dims.clone();
                    let extended_range = extend_to_dims[i] - 1;
                    extend_to_dims[i] = extended_range.next_multiple_of(data_window.dims()[i] - 1);

                    let mut extend_range = Vec::from(data_window.dims());
                    extend_range[i] -= 1;

                    let data_window_part = data_window.custom_range(&extend_range);
                    let extended_part = ResampleIScale.forwards(
                        data_window_part.as_block(),
                        ResampleCfg::new(&extend_to_dims),
                    );
                    let extended_part_window = extended_part.window();

                    extend_to_dims[i] = extended_range;
                    let extended_part_window =
                        extended_part_window.into_custom_range(&extend_to_dims);

                    let mut extended = VolumeBlock::new_zero(&extended_dims).unwrap();
                    let mut extended_window = extended.window_mut();

                    extended_part_window
                        .clone_to(&mut extended_window.custom_range_mut(&extend_to_dims));

                    let mut offset = vec![0; extended_window.dims().len()];
                    offset[i] = extend_to_dims[i] / 2;

                    let mut range = Vec::from(extended_window.dims());
                    range[i] = 1;

                    let data_window = data_window.custom_window(&offset, &range);

                    offset[i] = extend_to_dims[i];
                    data_window.clone_to(&mut extended_window.custom_window_mut(&offset, &range));

                    data = extended;
                }
            }
        }

        data
    }

    pub(crate) fn combined_blocks(&self, steps: &[u32]) -> Vec<usize> {
        let mut combined = self.dims.clone();

        for (&step, combined) in steps.iter().zip(&mut combined) {
            for _ in 0..step {
                *combined = (*combined / 2) + (*combined % 2);
            }
        }

        combined
    }
}

impl<Meta, T> GreedyTransformCoefficents<Meta, T> {
    /// Writes out the coefficients to the filesystem.
    pub fn write_out_partial(self, path: impl AsRef<Path>, compression: CompressionLevel)
    where
        T: Serializable,
    {
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
        stream.write_encode(compression, header_file).unwrap();

        for (i, level) in self.levels.into_iter().enumerate() {
            let output_path = path.join(format!("block_part_{i}.bin"));
            level.write_out_partial(output_path, compression);
        }
    }

    /// Reads in a subset of the data, reconstructible to
    /// the passed detail level, from the filesystem.
    pub fn read_for_steps(
        steps: &[u32],
        path: impl AsRef<Path>,
        filter: &impl DerivableMetadataFilter<Meta, T>,
    ) -> Self
    where
        T: Zero + Deserializable + Clone,
        Meta: Zero + Clone,
    {
        Self::read_for_steps_impl(steps, None, path, filter).0
    }

    pub(crate) fn read_for_steps_impl(
        steps: &[u32],
        non_greedy_size: Option<&[usize]>,
        path: impl AsRef<Path>,
        filter: &impl DerivableMetadataFilter<Meta, T>,
    ) -> (Self, Vec<u32>)
    where
        T: Zero + Deserializable + Clone,
        Meta: Zero + Clone,
    {
        let path = path.as_ref();
        if !path.exists() || !path.is_dir() {
            panic!("Invalid path {path:?}");
        }

        let f = std::fs::File::open(path.join("block_header.bin")).unwrap();
        let stream = DeserializeStream::new_decode(f).unwrap();
        let mut stream = stream.stream();

        let dims = Vec::<usize>::deserialize(&mut stream);
        let low = VolumeBlock::<T>::deserialize(&mut stream);
        let level_dims = Vec::<usize>::deserialize(&mut stream);

        if steps.len() != dims.len() {
            panic!("Invalid number of steps {steps:?} for volume of size {dims:?}");
        }

        let max_steps = dims
            .iter()
            .map(|&d| d.next_power_of_two().ilog2())
            .collect::<Vec<_>>();

        let steps = if let Some(non_greedy) = non_greedy_size {
            let steps_diff = max_steps
                .iter()
                .zip(non_greedy)
                .map(|(max, dims)| dims.ilog2() - max);

            steps
                .iter()
                .zip(steps_diff)
                .map(|(&s, diff)| if s < diff { 0 } else { s - diff })
                .collect::<Vec<_>>()
        } else {
            steps
                .iter()
                .zip(&max_steps)
                .map(|(&s, &max)| s.min(max))
                .collect::<Vec<_>>()
        };

        let (level_metas, meta) = Self::derive_metadata(&dims, &level_dims, filter);
        if steps.iter().all(|&s| s == 0) {
            return (
                Self {
                    dims: low.dims().into(),
                    low,
                    meta,
                    levels: vec![],
                },
                steps,
            );
        }

        let mut levels = Vec::new();
        let mut steps_skipped = max_steps
            .into_iter()
            .zip(&steps)
            .map(|(max, &s)| max - s)
            .collect::<Vec<_>>();
        for (i, (dim, meta)) in level_dims.into_iter().zip(level_metas).enumerate() {
            if steps_skipped[dim] > 0 {
                steps_skipped[dim] -= 1;
                continue;
            }

            let block_path = path.join(format!("block_part_{i}.bin"));
            let level = CoeffLevel::read_in_adapt(block_path, meta, &steps_skipped, filter);
            levels.push(level);
        }

        let dims = levels.first().unwrap().dims.clone();
        (
            Self {
                dims,
                low,
                meta,
                levels,
            },
            steps,
        )
    }

    /// Transforms a [`VolumeBlock`] containing the traditional wavelet
    /// transform into the greedy format.
    pub fn new_from_volume(
        volume: VolumeBlock<T>,
        decomposition: &[usize],
        filter: &impl DerivableMetadataFilter<Meta, T>,
    ) -> Self
    where
        T: Clone,
        Meta: Zero + Clone,
    {
        let dims = Vec::from(volume.dims());
        Self::new_from_volume_custom_size(volume, &dims, decomposition, filter)
    }

    pub(crate) fn new_from_volume_custom_size(
        volume: VolumeBlock<T>,
        dims: &[usize],
        decomposition: &[usize],
        filter: &impl DerivableMetadataFilter<Meta, T>,
    ) -> Self
    where
        T: Clone,
        Meta: Zero + Clone,
    {
        assert!(volume.dims().iter().all(|d| d.is_power_of_two()));
        assert!(dims.iter().all(|d| d.is_power_of_two()));
        assert!(volume.dims().iter().zip(dims).all(|(v, c)| v <= c));
        assert_eq!(volume.dims().len(), dims.len());

        let mut metadata = filter.derive_init(dims);

        fn adapt_metadata<Meta, T>(
            meta: VolumeWindow<'_, Meta>,
            filter: &impl DerivableMetadataFilter<Meta, T>,
            dim: usize,
        ) -> VolumeBlock<Meta>
        where
            Meta: Zero + Clone,
        {
            let mut dims = Vec::from(meta.dims());
            dims[dim] /= 2;

            let mut adapted = VolumeBlock::new_zero(&dims).unwrap();
            let mut adapted_window = adapted.window_mut();

            let meta_lanes = meta.lanes(dim);
            let adapted_lanes = adapted_window.lanes_mut(dim);

            for (meta, adapted) in meta_lanes.zip(adapted_lanes) {
                filter.derive_step(meta, adapted);
            }

            adapted
        }

        let adapt_steps = volume
            .dims()
            .iter()
            .zip(dims)
            .map(|(&v, &d)| d.ilog2() - v.ilog2());
        for (i, steps) in adapt_steps.enumerate() {
            for _ in 0..steps {
                let metadata_window = metadata.window();
                metadata = adapt_metadata(metadata_window, filter, i);
            }
        }

        let dims = Vec::from(volume.dims());
        let low = VolumeBlock::new_fill(&vec![1; dims.len()], volume[0].clone()).unwrap();
        let mut levels = Vec::new();

        let mut volume_window = volume.window();
        for &dim in decomposition {
            let dims = Vec::from(volume_window.dims());
            let (left, right) = volume_window.split_into(dim);

            let metadata_window = metadata.window();
            let meta = adapt_metadata(metadata_window, filter, dim);

            levels.push(CoeffLevel {
                dim,
                dims: dims.clone(),
                split: None,
                coeff: right.as_block(),
                metadata: meta.clone(),
            });

            volume_window = left;
            metadata = meta;
        }

        Self {
            dims,
            low,
            meta: metadata,
            levels,
        }
    }

    fn derive_metadata(
        dims: &[usize],
        steps: &[usize],
        filter: &impl DerivableMetadataFilter<Meta, T>,
    ) -> (Vec<VolumeBlock<Meta>>, VolumeBlock<Meta>)
    where
        Meta: Zero + Clone,
    {
        let mut metadata = filter.derive_init(dims);
        let mut level_meta = vec![];

        for &step in steps {
            let metadata_window = metadata.window();

            let mut aggregated_dims = Vec::from(metadata_window.dims());
            aggregated_dims[step] = (aggregated_dims[step] / 2) + (aggregated_dims[step] % 2);
            let aggregated_dims = aggregated_dims;

            let mut aggregation_range = Vec::from(metadata_window.dims());
            aggregation_range[step] /= 2;

            let (split, transform_range) = if metadata_window.dims()[step] % 2 == 0 {
                (None, Vec::from(metadata_window.dims()))
            } else {
                let mut range = Vec::from(metadata_window.dims());
                range[step] -= 1;

                (Some(metadata_window.dims()[step] - 1), range)
            };
            let transform_meta = metadata_window.custom_range(&transform_range);

            let mut aggreagated_meta = VolumeBlock::new_zero(&aggregated_dims).unwrap();
            let mut aggreagated_meta_window = aggreagated_meta.window_mut();
            let mut aggreagated_meta_window_part =
                aggreagated_meta_window.custom_range_mut(&aggregation_range);

            let meta_lanes = transform_meta.lanes(step);
            let aggregated_meta_lanes = aggreagated_meta_window_part.lanes_mut(step);
            for (meta, aggregated) in meta_lanes.zip(aggregated_meta_lanes) {
                filter.derive_step(meta, aggregated);
            }

            if let Some(split) = split {
                let mut offset = vec![0; transform_range.len()];
                offset[step] = split;

                let mut range = Vec::from(metadata.dims());
                range[step] = 1;

                let metadata_window_part = metadata_window.custom_window(&offset, &range);

                offset[step] = split / 2;
                metadata_window_part
                    .clone_to(&mut aggreagated_meta_window.custom_window_mut(&offset, &range));
            }

            metadata = aggreagated_meta.clone();
            level_meta.push(aggreagated_meta);
        }

        (level_meta, metadata)
    }
}

impl<Meta, T> CoeffLevel<Meta, T> {
    fn write_out_partial(self, path: impl AsRef<Path>, compression: CompressionLevel)
    where
        T: Serializable,
    {
        let mut stream = SerializeStream::new();
        self.dim.serialize(&mut stream);
        self.dims.serialize(&mut stream);
        self.split.serialize(&mut stream);
        self.coeff.serialize(&mut stream);

        let output_file = File::create(path).unwrap();
        stream.write_encode(compression, output_file).unwrap();
    }

    fn read_in_adapt(
        path: impl AsRef<Path>,
        meta: VolumeBlock<Meta>,
        skipped: &[u32],
        filter: &impl GreedyFilter<Meta, T>,
    ) -> Self
    where
        T: Zero + Deserializable + Clone,
        Meta: Zero + Clone,
    {
        let f = std::fs::File::open(path).unwrap();
        let stream = DeserializeStream::new_decode(f).unwrap();
        let mut stream = stream.stream();

        let dim = Deserializable::deserialize(&mut stream);
        let dims = Deserializable::deserialize(&mut stream);
        let split = Deserializable::deserialize(&mut stream);
        let coeff = Deserializable::deserialize(&mut stream);

        if skipped.iter().all(|&s| s == 0) {
            return Self {
                dim,
                dims,
                split,
                coeff,
                metadata: meta,
            };
        }

        let meta_window = meta.window();
        let meta = meta_window.custom_range(coeff.dims()).as_block();
        let adapted =
            GreedyTransformCoefficents::apply_forwards(coeff, meta, filter, skipped, true);

        let coeff = adapted.low;
        let meta = adapted.meta;

        let mut adapted_dims = Vec::from(coeff.dims());
        adapted_dims[dim] = dims[dim];
        let dims = adapted_dims;

        Self {
            dim,
            dims,
            split,
            coeff,
            metadata: meta,
        }
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
        self.meta.serialize(stream);
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
        let meta = Deserializable::deserialize(stream);
        let levels = Deserializable::deserialize(stream);

        Self {
            dims,
            low,
            meta,
            levels,
        }
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
        self.split.serialize(stream);
        self.coeff.serialize(stream);
        self.metadata.serialize(stream);
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
        let split = Deserializable::deserialize(stream);
        let coeff = Deserializable::deserialize(stream);
        let metadata = Deserializable::deserialize(stream);

        Self {
            dim,
            dims,
            split,
            coeff,
            metadata,
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
    let mut output_window = output.window_mut();

    let data_lanes = data.lanes(dim);
    let coeff_lanes = coeff.lanes(dim);
    let meta_lanes = meta.lanes(dim);

    let output_lanes = output_window.lanes_mut(dim);

    for (((data, coeff), meta), mut output) in data_lanes
        .zip(coeff_lanes)
        .zip(meta_lanes)
        .zip(output_lanes)
    {
        for (src, dst) in data.iter().zip(data_low.iter_mut()) {
            *dst = src.clone();
        }
        for (src, dst) in coeff.iter().zip(data_high.iter_mut()) {
            *dst = src.clone();
        }
        for (src, dst) in meta.iter().zip(meta_scratch.iter_mut()) {
            *dst = src.clone();
        }

        f.backwards(&mut output, data_low, data_high, &meta_scratch);
    }

    output
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
