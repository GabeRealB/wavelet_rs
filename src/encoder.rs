//! Utilities for encoding a large dataset.

use std::{
    borrow::Borrow, collections::BTreeMap, fmt::Debug, fs::File, marker::PhantomData, path::Path,
};

use num_traits::Zero;

use crate::{
    filter::{Filter, GenericFilter, ToGenericFilter},
    range::{for_each_range, for_each_range_enumerate, for_each_range_par_enumerate},
    stream::{AnyMap, Deserializable, DeserializeStream, Serializable, SerializeStream},
    transformations::{
        wavelet_transform::BackwardsOperation, Chain, ResampleCfg, ResampleExtend,
        ReversibleTransform, WaveletDecompCfg, WaveletTransform,
    },
    utilities::{flatten_idx, flatten_idx_unchecked, strides_for_dims},
    volume::{VolumeBlock, VolumeWindowMut},
};

/// Encoder of a dataset with the wavelet transform.
pub struct VolumeWaveletEncoder<'a, T> {
    metadata: AnyMap,
    dims: Vec<usize>,
    strides: Vec<usize>,
    num_base_dims: usize,
    fetchers: Vec<Option<VolumeFetcher<'a, T>>>,
}

pub(crate) struct OutputHeader<T> {
    pub dims: Vec<usize>,
    pub metadata: AnyMap,
    pub block_size: Vec<usize>,
    pub block_counts: Vec<usize>,
    pub input_block_dims: Vec<usize>,
    pub block_blueprints: BlockBlueprints<T>,
    pub filter: GenericFilter<T>,
}

impl OutputHeader<()> {
    pub fn deserialize_info(
        stream: &mut crate::stream::DeserializeStreamRef<'_>,
    ) -> (String, Vec<usize>) {
        let t_name: String = Deserializable::deserialize(stream);

        let dims = Deserializable::deserialize(stream);
        (t_name, dims)
    }
}

impl<T: Serializable> Serializable for OutputHeader<T> {
    fn serialize(self, stream: &mut SerializeStream) {
        std::any::type_name::<T>().serialize(stream);

        self.dims.serialize(stream);
        self.metadata.serialize(stream);
        self.block_size.serialize(stream);
        self.block_counts.serialize(stream);
        self.input_block_dims.serialize(stream);
        self.block_blueprints.serialize(stream);
        self.filter.serialize(stream);
    }
}

impl<T: Deserializable> Deserializable for OutputHeader<T> {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let t_name: String = Deserializable::deserialize(stream);
        assert_eq!(t_name, std::any::type_name::<T>());

        let dims = Deserializable::deserialize(stream);
        let metadata = Deserializable::deserialize(stream);
        let block_size = Deserializable::deserialize(stream);
        let block_counts = Deserializable::deserialize(stream);
        let input_block_dims = Deserializable::deserialize(stream);
        let block_blueprints = Deserializable::deserialize(stream);
        let filter = Deserializable::deserialize(stream);

        Self {
            dims,
            metadata,
            block_size,
            block_counts,
            input_block_dims,
            block_blueprints,
            filter,
        }
    }
}

type VolumeFetcher<'a, T> = Box<dyn Fn(&[usize]) -> T + Sync + Send + 'a>;

impl<'a, T> VolumeWaveletEncoder<'a, T>
where
    T: Zero + Serializable + Send + Clone,
{
    /// Constructs a new encoder over a volume of dimensionality `dims`.
    ///
    /// The parameter `num_base_dims` describes the number of dimensions
    /// contained in each fetcher.
    pub fn new(dims: &[usize], num_base_dims: usize) -> Self {
        assert!(dims.len() > num_base_dims);
        let num_fetchers = dims[num_base_dims..].iter().product();
        let mut fetchers = Vec::with_capacity(num_fetchers);
        for _ in 0..num_fetchers {
            fetchers.push(None);
        }

        let strides = strides_for_dims(&dims[num_base_dims..]);
        Self {
            metadata: AnyMap::new(),
            dims: dims.into(),
            strides,
            num_base_dims,
            fetchers,
        }
    }

    /// Adds a closure to the list of data fetchers of the encoder.
    pub fn add_fetcher(&mut self, index: &[usize], f: impl Fn(&[usize]) -> T + Sync + Send + 'a) {
        let idx = flatten_idx(&self.dims[self.num_base_dims..], &self.strides, index);
        assert!(self.fetchers[idx].is_none());
        self.fetchers[idx] = Some(Box::new(f));
    }

    /// Fetches a value inserted into the metadata.
    pub fn get_metadata<Q, M>(&self, key: &Q) -> Option<M>
    where
        String: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
        M: Deserializable,
    {
        self.metadata.get(key)
    }

    /// Inserts some metadata which will be included into the encoded dataset.
    pub fn insert_metadata<M>(&mut self, key: String, value: M) -> bool
    where
        M: Serializable,
    {
        self.metadata.insert(key, value)
    }

    /// Encodes the dataset with the specified block size and filter.
    pub fn encode(
        &self,
        output: impl AsRef<Path> + Sync,
        block_size: &[usize],
        filter: impl ToGenericFilter<T> + Serializable + Clone,
    ) where
        GenericFilter<T>: Filter<T>,
    {
        assert_eq!(block_size.len(), self.dims.len());
        assert!(block_size
            .iter()
            .zip(&self.dims)
            .all(|(&block, &dim)| dim % block == 0 && block <= dim));

        let filter = filter.to_generic();

        let block_resample_dims: Vec<_> = block_size
            .iter()
            .map(|size| size.next_power_of_two())
            .collect();
        let steps: Vec<_> = block_resample_dims
            .iter()
            .map(|size| size.ilog2())
            .collect();
        let block_transform_cfg = Chain::from((
            ResampleCfg::new(&block_resample_dims),
            WaveletDecompCfg::new(&steps),
        ));
        let block_transform =
            Chain::from((ResampleExtend, WaveletTransform::new(filter.clone(), false)));

        let block_counts: Vec<_> = block_size
            .iter()
            .zip(&self.dims)
            .map(|(&block, &dim)| dim / block)
            .collect();
        let block_counts_range: Vec<_> = block_counts.iter().map(|&c| 0..c).collect();
        let block_range: Vec<_> = block_size.iter().map(|&b| 0..b).collect();

        if !output.as_ref().exists() {
            std::fs::create_dir(output.as_ref()).unwrap();
        }

        let (sx, rx) = std::sync::mpsc::sync_channel(block_counts.iter().product());
        for_each_range_par_enumerate(block_counts_range.iter().cloned(), |i, block_idx| {
            let block_offset: Vec<_> = block_idx
                .iter()
                .zip(block_size)
                .map(|(&idx, &size)| idx * size)
                .collect();

            let mut block = VolumeBlock::new_zero(block_size).unwrap();

            for_each_range_enumerate(block_range.iter().cloned(), |i, inner_idx| {
                let idx: Vec<_> = block_offset
                    .iter()
                    .zip(inner_idx)
                    .map(|(&offset, &idx)| offset + idx)
                    .collect();
                let fetcher_idx = unsafe { self.flatten_idx_full_unchecked(&idx) };
                let fetcher = self.fetchers[fetcher_idx].as_ref().unwrap();
                block[i] = fetcher(&idx[0..self.num_base_dims]);
            });

            let block_decomp = block_transform.forwards(block, block_transform_cfg);
            sx.send((i, block_decomp[0].clone())).unwrap();

            let block_dir = output.as_ref().join(format!("block_{i}"));
            if block_dir.exists() {
                std::fs::remove_dir_all(&block_dir).unwrap();
            }
            std::fs::create_dir(&block_dir).unwrap();

            let mut counter = 0;
            let mut block_decomp_window = block_decomp.window();
            while block_decomp_window.dims().iter().any(|&d| d != 1) {
                let num_dims = block_decomp_window.dims().len();

                for dim in 0..num_dims {
                    if block_decomp_window.dims()[dim] == 1 {
                        continue;
                    }

                    let (low, high) = block_decomp_window.split_into(dim);
                    block_decomp_window = low;

                    let lanes = high.lanes(0);
                    let mut stream = SerializeStream::new();
                    for lane in lanes {
                        for elem in lane.as_slice().unwrap() {
                            elem.clone().serialize(&mut stream);
                        }
                    }

                    let out_path = block_dir.join(format!("block_part_{counter}.bin"));
                    let out_file = File::create(out_path).unwrap();
                    stream.write_encode(out_file).unwrap();

                    counter += 1;
                }
            }
        });

        let mut superblock = VolumeBlock::new_zero(&block_counts).unwrap();
        for (i, elem) in rx.try_iter() {
            superblock[i] = elem;
        }

        let resample_dims: Vec<_> = block_counts
            .iter()
            .map(|size| size.next_power_of_two())
            .collect();
        let steps: Vec<_> = resample_dims.iter().map(|size| size.ilog2()).collect();

        let output_transform_cfg = Chain::from((
            ResampleCfg::new(&resample_dims),
            WaveletDecompCfg::new(&steps),
        ));
        let output_transform =
            Chain::from((ResampleExtend, WaveletTransform::new(filter.clone(), false)));
        let transformed = output_transform.forwards(superblock, output_transform_cfg);

        let mut stream = SerializeStream::new();

        let output_header = OutputHeader {
            metadata: self.metadata.clone(),
            dims: self.dims.clone(),
            block_size: block_size.into(),
            block_counts,
            input_block_dims: transformed.dims().into(),
            block_blueprints: BlockBlueprints::<T>::new(&block_resample_dims),
            filter,
        };
        output_header.serialize(&mut stream);
        for elem in transformed.flatten() {
            elem.clone().serialize(&mut stream);
        }
        let output_path = output.as_ref().join("output.bin");
        let output_file = File::create(output_path).unwrap();
        stream.write_encode(output_file).unwrap();
    }

    unsafe fn flatten_idx_full_unchecked(&self, index: &[usize]) -> usize {
        flatten_idx_unchecked(&self.strides, &index[self.num_base_dims..])
    }
}

impl<'a, T> Debug for VolumeWaveletEncoder<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VolumeWaveletEncoder")
            .field("metadata", &self.metadata)
            .field("dims", &self.dims)
            .field("strides", &self.strides)
            .field("num_base_dims", &self.num_base_dims)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BlockBlueprints<T> {
    dims: usize,
    layouts: Vec<BlockLayout>,
    blueprints: BTreeMap<Vec<u32>, BlockBlueprint<T>>,
}

impl<T> BlockBlueprints<T> {
    pub fn new(dims: &[usize]) -> Self {
        let mut layouts = Vec::new();
        let mut size: Vec<_> = dims.into();
        while size.iter().any(|&s| s > 1) {
            for dim in 0..size.len() {
                if size[dim] > 1 {
                    size[dim] /= 2;

                    let mut offset = vec![0; size.len()];
                    offset[dim] = size[dim];

                    layouts.push(BlockLayout {
                        dim,
                        size: size.clone(),
                        offset,
                    })
                }
            }
        }

        let mut blueprints = BTreeMap::new();
        let steps_range = dims.iter().rev().map(|&d| 0..(d.ilog2() + 1) as usize);

        for_each_range(steps_range, |refined| {
            if refined.iter().all(|&r| r == 0) {
                blueprints.insert(vec![0; dims.len()], BlockBlueprint::new(dims));
            } else {
                let steps: Vec<_> = refined.iter().rev().map(|&r| r as u32).collect();
                let b = BlockBlueprint::new_refined(&steps, &layouts, &blueprints);
                blueprints.insert(steps, b);
            }
        });

        Self {
            dims: dims.len(),
            layouts,
            blueprints,
        }
    }

    pub fn reconstruct_full(
        &self,
        filter: &(impl Filter<T> + Clone),
        block_path: impl AsRef<Path>,
        steps: &[u32],
    ) -> VolumeBlock<T>
    where
        T: Zero + Deserializable + Send + Clone,
    {
        assert_eq!(self.dims, steps.len());

        for (k, b) in &self.blueprints {
            if k.iter().all(|&s| s == 0) {
                return b.reconstruct(filter, &self.layouts, block_path, steps);
            }
        }

        unreachable!()
    }

    pub fn reconstruct(
        &self,
        filter: &(impl Filter<T> + Clone),
        block_path: impl AsRef<Path>,
        steps: &[u32],
        refinements: &[u32],
    ) -> VolumeBlock<T>
    where
        T: Zero + Deserializable + Send + Clone,
    {
        assert_eq!(self.dims, steps.len());
        assert_eq!(self.dims, refinements.len());

        let blueprint = self.blueprints.get(steps).unwrap();
        blueprint.reconstruct(filter, &self.layouts, block_path, refinements)
    }

    pub fn block_decompositions_full(&self, steps: &[u32]) -> Vec<u32> {
        assert_eq!(self.dims, steps.len());

        for (k, b) in &self.blueprints {
            if k.iter().all(|&s| s == 0) {
                return b.block_decompositions(&self.layouts, steps);
            }
        }

        unreachable!()
    }

    pub fn start_dim_full(&self, steps: &[u32]) -> usize {
        assert_eq!(self.dims, steps.len());

        for (k, b) in &self.blueprints {
            if k.iter().all(|&s| s == 0) {
                return b.start_dim(&self.layouts, steps);
            }
        }

        unreachable!()
    }

    pub fn start_dim(&self, steps: &[u32], refinements: &[u32]) -> usize {
        assert_eq!(self.dims, steps.len());
        assert_eq!(self.dims, refinements.len());

        let blueprint = self.blueprints.get(steps).unwrap();
        blueprint.start_dim(&self.layouts, refinements)
    }
}

impl<T: Serializable> Serializable for BlockBlueprints<T> {
    fn serialize(self, stream: &mut SerializeStream) {
        self.dims.serialize(stream);
        self.layouts.serialize(stream);
        self.blueprints.serialize(stream);
    }
}

impl<T: Deserializable> Deserializable for BlockBlueprints<T> {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let dims = Deserializable::deserialize(stream);
        let layouts = Deserializable::deserialize(stream);
        let blueprints = Deserializable::deserialize(stream);

        Self {
            dims,
            layouts,
            blueprints,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct BlockBlueprint<T> {
    base_size: Vec<usize>,
    parts: Vec<BlockBlueprintPart>,
    _phantom: PhantomData<fn() -> T>,
}

impl<T> BlockBlueprint<T> {
    fn new(size: &[usize]) -> Self {
        let base_size = vec![1; size.len()];
        let mut curr: Vec<_> = size.into();

        let mut id = 0;
        let mut parts = vec![];

        while curr.iter().any(|&c| c > 1) {
            for size in &mut curr {
                if *size <= 1 {
                    continue;
                }

                parts.push(BlockBlueprintPart {
                    id,
                    adapted_size: None,
                    adapt: None,
                });

                *size /= 2;
                id += 1;
            }
        }

        Self {
            base_size,
            parts,
            _phantom: PhantomData,
        }
    }

    fn new_refined(
        steps: &[u32],
        blocks: &[BlockLayout],
        blueprints: &BTreeMap<Vec<u32>, Self>,
    ) -> Self {
        let mut prev: Vec<_> = steps.into();
        let mut dim = None;
        for (i, x) in prev.iter_mut().enumerate() {
            if *x > 0 {
                dim = Some(i);
                *x -= 1;
                break;
            }
        }
        let dim = dim.unwrap();

        let template = &blueprints[&prev];

        let mut part_idx = None;
        let mut counter = 0;
        for (i, part) in template.parts.iter().rev().enumerate() {
            if blocks[part.id].dim == dim {
                counter += 1;
                if counter == steps[dim] - prev[dim] {
                    part_idx = Some(template.parts.len() - i - 1);
                    break;
                }
            }
        }
        let part_idx = part_idx.unwrap();
        let part_to_remove = &template.parts[part_idx];
        let part_to_remove_block = &blocks[part_to_remove.id];

        let mut base_size = template.base_size.clone();
        base_size[dim] *= 2;

        let mut parts = template.parts.clone();
        parts.remove(part_idx);
        for part in &mut parts[part_idx..] {
            let mut adapted_size = part
                .adapted_size
                .take()
                .unwrap_or_else(|| blocks[part.id].size.clone());
            let mut adapt = part.adapt.take().unwrap_or_default();

            let required_steps = adapted_size
                .iter()
                .zip(&part_to_remove_block.size)
                .map(|(&r, &o)| (o / r).max(1).ilog2())
                .collect();

            adapted_size[part_to_remove_block.dim] *= 2;
            adapt.push((part_to_remove.id, required_steps));

            part.adapted_size = Some(adapted_size);
            part.adapt = Some(adapt);
        }

        Self {
            base_size,
            parts,
            _phantom: PhantomData,
        }
    }

    fn block_size(&self, blocks: &[BlockLayout], steps: &[u32]) -> Vec<usize> {
        if steps.iter().all(|&s| s == 0) {
            self.base_size.clone()
        } else {
            let mut curr = vec![0; steps.len()];

            for part in self.parts.iter().rev() {
                curr[blocks[part.id].dim] += 1;

                if steps.iter().zip(&curr).all(|(&s, &c)| c >= s) {
                    break;
                }
            }

            curr.iter()
                .zip(&self.base_size)
                .map(|(&st, &si)| (si << st))
                .collect()
        }
    }

    fn block_decompositions(&self, blocks: &[BlockLayout], steps: &[u32]) -> Vec<u32> {
        if steps.iter().all(|&s| s == 0) {
            vec![0; steps.len()]
        } else {
            let mut curr = vec![0; steps.len()];

            for part in self.parts.iter().rev() {
                curr[blocks[part.id].dim] += 1;

                if steps.iter().zip(&curr).all(|(&s, &c)| c >= s) {
                    break;
                }
            }

            curr
        }
    }

    fn start_dim(&self, blocks: &[BlockLayout], steps: &[u32]) -> usize {
        if steps.iter().all(|&s| s == 0) {
            0
        } else {
            let mut curr = vec![0; steps.len()];

            for part in self.parts.iter().rev() {
                curr[blocks[part.id].dim] += 1;

                if steps.iter().zip(&curr).all(|(&s, &c)| c >= s) {
                    return blocks[part.id].dim;
                }
            }

            0
        }
    }

    fn reconstruct(
        &self,
        filter: &(impl Filter<T> + Clone),
        blocks: &[BlockLayout],
        block_path: impl AsRef<Path>,
        steps: &[u32],
    ) -> VolumeBlock<T>
    where
        T: Zero + Deserializable + Send + Clone,
    {
        assert_eq!(steps.len(), self.base_size.len());

        let block_size = self.block_size(blocks, steps);
        let mut block = VolumeBlock::new_zero(&block_size).unwrap();

        if steps.iter().all(|&s| s == 0) {
            return block;
        }

        let mut curr = vec![0; steps.len()];

        let mut block_window = block.window_mut();

        let mut cache = Some(BTreeMap::new());
        for part in self.parts.iter().rev() {
            if curr[blocks[part.id].dim] < steps[blocks[part.id].dim] {
                curr[blocks[part.id].dim] += 1;
                part.insert_in_block(filter, &block_path, &mut block_window, blocks, &mut cache);

                if curr == steps {
                    break;
                }
            }
        }

        block
    }
}

impl<T: Serializable> Serializable for BlockBlueprint<T> {
    fn serialize(self, stream: &mut SerializeStream) {
        self.base_size.serialize(stream);
        self.parts.serialize(stream);
    }
}

impl<T: Deserializable> Deserializable for BlockBlueprint<T> {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let base_size = Deserializable::deserialize(stream);
        let parts = Deserializable::deserialize(stream);

        Self {
            base_size,
            parts,
            _phantom: PhantomData,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct BlockLayout {
    dim: usize,
    size: Vec<usize>,
    offset: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct BlockBlueprintPart {
    id: usize,
    adapted_size: Option<Vec<usize>>,
    adapt: Option<Vec<(usize, Vec<u32>)>>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum CoeffType {
    Low,
    High,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CacheKey {
    id: usize,
    decomp: Vec<u32>,
    coeff_type: CoeffType,
}

impl BlockBlueprintPart {
    fn init_cache<T>(
        filter: &(impl Filter<T> + Clone),
        block_path: impl AsRef<Path>,
        size: Vec<usize>,
        part_id: usize,
        steps: Vec<u32>,
        cache: &mut BTreeMap<CacheKey, VolumeBlock<T>>,
    ) where
        T: Zero + Deserializable + Send + Clone,
    {
        let k = CacheKey {
            id: part_id,
            decomp: steps,
            coeff_type: CoeffType::Low,
        };
        if cache.contains_key(&k) {
            return;
        }

        let steps = k.decomp;
        if steps.iter().all(|&s| s == 0) {
            let block_path = block_path
                .as_ref()
                .join(format!("block_part_{}.bin", part_id));

            let f = std::fs::File::open(block_path).unwrap();
            let stream = DeserializeStream::new_decode(f).unwrap();
            let mut stream = stream.stream();

            let mut block_part = VolumeBlock::new_zero(&size).unwrap();
            let mut block_part_window = block_part.window_mut();
            let lanes = block_part_window.lanes_mut(0);
            for mut lane in lanes {
                for elem in lane.as_slice_mut().unwrap() {
                    *elem = Deserializable::deserialize(&mut stream);
                }
            }

            let entry = cache.entry(CacheKey {
                id: part_id,
                decomp: steps,
                coeff_type: CoeffType::Low,
            });
            entry.or_insert(block_part);
        } else {
            let mut prev_size = size.clone();
            let mut prev_steps = steps.clone();
            let mut dim = None;
            for (i, x) in prev_steps.iter_mut().enumerate() {
                if *x > 0 {
                    *x -= 1;
                    prev_size[i] *= 2;
                    dim = Some(i);
                    break;
                }
            }
            let dim = dim.unwrap();

            BlockBlueprintPart::init_cache(
                filter,
                block_path,
                prev_size,
                part_id,
                prev_steps.clone(),
                cache,
            );

            let prev_key = CacheKey {
                id: part_id,
                decomp: prev_steps,
                coeff_type: CoeffType::Low,
            };
            let prev = &cache[&prev_key];

            let mut rec_steps = vec![0; size.len()];
            rec_steps[dim] = 1;

            let cfg = WaveletDecompCfg::new(&rec_steps);
            let w = WaveletTransform::<T, _>::new(filter.clone(), false);
            let part = w.forwards(prev.clone(), cfg);

            let part_window = part.window();
            let (low, high) = part_window.split_into(dim);

            let mut block_part = VolumeBlock::new_zero(&size).unwrap();
            low.clone_to(&mut block_part.window_mut());
            let entry = cache.entry(CacheKey {
                id: part_id,
                decomp: steps.clone(),
                coeff_type: CoeffType::Low,
            });
            entry.or_insert(block_part);

            let mut block_part = VolumeBlock::new_zero(&size).unwrap();
            high.clone_to(&mut block_part.window_mut());
            let entry = cache.entry(CacheKey {
                id: part_id,
                decomp: steps,
                coeff_type: CoeffType::High,
            });
            entry.or_insert(block_part);
        }
    }

    #[allow(clippy::type_complexity)]
    fn insert_in_block<T>(
        &self,
        filter: &(impl Filter<T> + Clone),
        block_path: impl AsRef<Path>,
        block: &mut VolumeWindowMut<'_, T>,
        blocks: &[BlockLayout],
        cache: &mut Option<BTreeMap<CacheKey, VolumeBlock<T>>>,
    ) where
        T: Zero + Deserializable + Send + Clone,
    {
        if let Some(cache) = cache {
            if let Some((adapted_size, adapt)) = self.adapted_size.as_ref().zip(self.adapt.as_ref())
            {
                let mut decomp = VolumeBlock::new_zero(adapted_size).unwrap();
                let mut decomp_window = decomp.window_mut();

                {
                    let steps = vec![0; blocks[self.id].size.len()];
                    BlockBlueprintPart::init_cache(
                        filter,
                        &block_path,
                        blocks[self.id].size.clone(),
                        self.id,
                        steps.clone(),
                        cache,
                    );

                    let block_part_key = CacheKey {
                        id: self.id,
                        decomp: steps,
                        coeff_type: CoeffType::Low,
                    };
                    let block_part = cache.get(&block_part_key).unwrap();
                    let block_part_window = block_part.window();

                    let mut decomp_window = decomp_window.custom_range_mut(&blocks[self.id].size);
                    block_part_window.clone_to(&mut decomp_window);
                }

                let mut ops = Vec::with_capacity(adapt.len());
                for (id, steps) in adapt.iter().rev() {
                    let part_size: Vec<_> = blocks[*id]
                        .size
                        .iter()
                        .zip(steps)
                        .map(|(&si, &st)| si >> st)
                        .collect();

                    BlockBlueprintPart::init_cache(
                        filter,
                        &block_path,
                        part_size.clone(),
                        *id,
                        steps.clone(),
                        cache,
                    );

                    let block_part_key = CacheKey {
                        id: *id,
                        decomp: steps.clone(),
                        coeff_type: CoeffType::High,
                    };
                    let block_part = cache.get(&block_part_key).unwrap();
                    let block_part_window = block_part.window();

                    let mut decomp_window =
                        decomp_window.custom_window_mut(&blocks[*id].offset, &part_size);
                    block_part_window.clone_to(&mut decomp_window);

                    ops.push(BackwardsOperation::Backwards {
                        dim: blocks[*id].dim,
                        adapt: None,
                    })
                }

                let mut scratch = vec![T::zero(); *decomp_window.dims().iter().max().unwrap()];
                let trans = WaveletTransform::<T, _>::new(filter.clone(), false);
                trans.back_(decomp_window, &mut scratch, &ops);

                let mut window = block.custom_window_mut(&blocks[self.id].offset, adapted_size);
                decomp.window().clone_to(&mut window);
            } else {
                let steps = vec![0; blocks[self.id].size.len()];
                BlockBlueprintPart::init_cache(
                    filter,
                    block_path,
                    blocks[self.id].size.clone(),
                    self.id,
                    steps.clone(),
                    cache,
                );

                let block_part_key = CacheKey {
                    id: self.id,
                    decomp: steps,
                    coeff_type: CoeffType::Low,
                };
                let block_part = cache.get(&block_part_key).unwrap();
                let block_part_window = block_part.window();

                let mut window =
                    block.custom_window_mut(&blocks[self.id].offset, &blocks[self.id].size);
                block_part_window.clone_to(&mut window);
            }
        } else {
            let block_path = block_path
                .as_ref()
                .join(format!("block_part_{}.bin", self.id));

            let f = std::fs::File::open(block_path).unwrap();
            let stream = DeserializeStream::new_decode(f).unwrap();
            let mut stream = stream.stream();

            let mut window =
                block.custom_window_mut(&blocks[self.id].offset, &blocks[self.id].size);
            let lanes = window.lanes_mut(0);
            for mut lane in lanes {
                for elem in lane.as_slice_mut().unwrap() {
                    *elem = Deserializable::deserialize(&mut stream);
                }
            }
        }
    }
}

impl Serializable for BlockLayout {
    fn serialize(self, stream: &mut SerializeStream) {
        self.dim.serialize(stream);
        self.size.serialize(stream);
        self.offset.serialize(stream);
    }
}

impl Deserializable for BlockLayout {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let dim = Deserializable::deserialize(stream);
        let size = Deserializable::deserialize(stream);
        let offset = Deserializable::deserialize(stream);

        Self { dim, size, offset }
    }
}

impl Serializable for BlockBlueprintPart {
    fn serialize(self, stream: &mut SerializeStream) {
        self.id.serialize(stream);
        self.adapted_size.serialize(stream);
        self.adapt.serialize(stream);
    }
}

impl Deserializable for BlockBlueprintPart {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let id = Deserializable::deserialize(stream);
        let adapted_size = Deserializable::deserialize(stream);
        let adapt = Deserializable::deserialize(stream);

        Self {
            id,
            adapted_size,
            adapt,
        }
    }
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::BufReader, path::PathBuf};

    use crate::{
        filter::AverageFilter, utilities::flatten_idx_unchecked, vector::Vector,
        volume::VolumeBlock,
    };

    use super::VolumeWaveletEncoder;

    #[test]
    fn encode() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test/encode");

        let dims = [256, 256, 256, 2];
        let mut encoder = VolumeWaveletEncoder::new(&dims, 3);

        let dims = [256, 256, 256];
        let num_elements = dims.iter().product();
        let block_1 = VolumeBlock::new_with_data(&dims, vec![1.0f32; num_elements]).unwrap();
        let fetcher_1 = move |idx: &[usize]| block_1[idx];
        let block_2 = VolumeBlock::new_with_data(&dims, vec![2.0f32; num_elements]).unwrap();
        let fetcher_2 = move |idx: &[usize]| block_2[idx];

        encoder.add_fetcher(&[0], fetcher_1);
        encoder.add_fetcher(&[1], fetcher_2);

        encoder.insert_metadata::<String>("Test".into(), "Example metadata string".into());

        let block_size = [32, 32, 32, 2];
        encoder.encode(res_path, &block_size, AverageFilter)
    }

    #[test]
    fn encode_sample() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test/encode_sample");

        let dims = [4, 4, 1];
        let mut encoder = VolumeWaveletEncoder::new(&dims, 2);

        let dims = [4, 4];
        let block = VolumeBlock::new_with_data(
            &dims,
            vec![
                1.0f32, 2.0f32, 3.0f32, 4.0f32, 6.0f32, 8.0f32, 10.0f32, 12.0f32, 15.0f32, 18.0f32,
                21.0f32, 24.0f32, 28.0f32, 32.0f32, 36.0f32, 40.0f32,
            ],
        )
        .unwrap();
        let fetcher = move |idx: &[usize]| block[idx];

        encoder.add_fetcher(&[0], fetcher);

        let block_size = [4, 4, 1];
        encoder.encode(res_path, &block_size, AverageFilter)
    }

    #[test]
    fn encode_img_1() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let img_path = res_path.join("resources/test/img_1.jpg");
        res_path.push("resources/test/encode_img_1");

        let f = File::open(img_path).unwrap();
        let reader = BufReader::new(f);
        let img = image::load(reader, image::ImageFormat::Jpeg)
            .unwrap()
            .to_rgb32f();

        let (width, height) = (img.width() as usize, img.height() as usize);
        let data: Vec<_> = img.pixels().map(|p| Vector::new(p.0)).collect();
        let fetcher = move |idx: &[usize]| {
            let idx = unsafe { flatten_idx_unchecked(&[1, width], idx) };
            data[idx]
        };

        let dims = [width, height, 1];
        let mut encoder = VolumeWaveletEncoder::new(&dims, 2);
        encoder.add_fetcher(&[0], fetcher);

        let block_size = [256, 256, 1];
        encoder.encode(res_path, &block_size, AverageFilter)
    }

    #[test]
    fn encode_img_2() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let img_path = res_path.join("resources/test/img_2.jpg");
        res_path.push("resources/test/encode_img_2");

        let f = File::open(img_path).unwrap();
        let reader = BufReader::new(f);
        let img = image::load(reader, image::ImageFormat::Jpeg)
            .unwrap()
            .to_rgb32f();

        let (width, height) = (img.width() as usize, img.height() as usize);
        let data: Vec<_> = img.pixels().map(|p| Vector::new(p.0)).collect();
        let fetcher = move |idx: &[usize]| {
            let idx = unsafe { flatten_idx_unchecked(&[1, width], idx) };
            data[idx]
        };

        let dims = [width, height, 1];
        let mut encoder = VolumeWaveletEncoder::new(&dims, 2);
        encoder.add_fetcher(&[0], fetcher);

        let block_size = [252, 252, 1];
        encoder.encode(res_path, &block_size, AverageFilter)
    }
}
