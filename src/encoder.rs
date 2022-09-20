use std::{
    borrow::Borrow, collections::BTreeMap, fmt::Debug, fs::File, marker::PhantomData, path::Path,
};

use num_traits::Num;

use crate::{
    filter::Filter,
    range::{for_each_range_enumerate, for_each_range_par_enumerate},
    stream::{AnyMap, Deserializable, DeserializeStream, Serializable, SerializeStream},
    transformations::{
        Chain, ResampleCfg, ResampleExtend, ReversibleTransform, WaveletDecompCfg, WaveletTransform,
    },
    utilities::{flatten_idx, flatten_idx_unchecked},
    volume::VolumeBlock,
};

pub struct VolumeWaveletEncoder<'a, T: Num + Copy> {
    metadata: AnyMap,
    dims: Vec<usize>,
    strides: Vec<usize>,
    num_base_dims: usize,
    fetchers: Vec<Option<VolumeFetcher<'a, T>>>,
}

pub(crate) struct OutputHeader<T, F> {
    pub num_type: String,
    pub metadata: AnyMap,
    pub dims: Vec<usize>,
    pub block_size: Vec<usize>,
    pub block_counts: Vec<usize>,
    pub input_block_dims: Vec<usize>,
    pub block_blueprints: BlockBlueprints<T>,
    pub filter: F,
}

impl<T: Serializable, F: Serializable> Serializable for OutputHeader<T, F> {
    fn serialize(self, stream: &mut SerializeStream) {
        T::name().serialize(stream);
        F::name().serialize(stream);

        self.num_type.serialize(stream);
        self.metadata.serialize(stream);
        self.dims.serialize(stream);
        self.block_size.serialize(stream);
        self.block_counts.serialize(stream);
        self.input_block_dims.serialize(stream);
        self.block_blueprints.serialize(stream);
        self.filter.serialize(stream);
    }
}

impl<T: Deserializable, F: Deserializable> Deserializable for OutputHeader<T, F> {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let t_name: String = Deserializable::deserialize(stream);
        assert_eq!(t_name, T::name());
        let f_name: String = Deserializable::deserialize(stream);
        assert_eq!(f_name, F::name());

        let num_type = Deserializable::deserialize(stream);
        let metadata = Deserializable::deserialize(stream);
        let dims = Deserializable::deserialize(stream);
        let block_size = Deserializable::deserialize(stream);
        let block_counts = Deserializable::deserialize(stream);
        let input_block_dims = Deserializable::deserialize(stream);
        let block_blueprints = Deserializable::deserialize(stream);
        let filter = Deserializable::deserialize(stream);

        Self {
            num_type,
            metadata,
            dims,
            block_size,
            block_counts,
            input_block_dims,
            block_blueprints,
            filter,
        }
    }
}

type VolumeFetcher<'a, T> = Box<dyn Fn(&[usize]) -> T + Sync + Send + 'a>;

impl<'a, T: Serializable + Num + Send + Copy> VolumeWaveletEncoder<'a, T> {
    pub fn new(dims: &[usize], num_base_dims: usize) -> Self {
        assert!(dims.len() > num_base_dims);
        let num_fetchers = dims[num_base_dims..].iter().product();
        let mut fetchers = Vec::with_capacity(num_fetchers);
        for _ in 0..num_fetchers {
            fetchers.push(None);
        }

        let strides = std::iter::once(1)
            .chain(dims[num_base_dims..].iter().scan(1usize, |s, &d| {
                *s *= d;
                Some(*s)
            }))
            .take(dims.len() - num_base_dims)
            .collect();

        Self {
            metadata: AnyMap::new(),
            dims: dims.into(),
            strides,
            num_base_dims,
            fetchers,
        }
    }

    pub fn add_fetcher(&mut self, index: &[usize], f: impl Fn(&[usize]) -> T + Sync + Send + 'a) {
        let idx = flatten_idx(&self.dims[self.num_base_dims..], &self.strides, index);
        assert!(self.fetchers[idx].is_none());
        self.fetchers[idx] = Some(Box::new(f));
    }

    pub fn get_metadata<Q, M>(&self, key: &Q) -> Option<M>
    where
        String: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
        M: Deserializable,
    {
        self.metadata.get(key)
    }

    pub fn insert_metadata<M>(&mut self, key: String, value: M) -> bool
    where
        M: Serializable,
    {
        self.metadata.insert(key, value)
    }

    pub fn encode(
        &self,
        output: impl AsRef<Path> + Sync,
        block_size: &[usize],
        filter: impl Filter<T> + Serializable + Clone,
    ) {
        assert_eq!(block_size.len(), self.dims.len());
        assert!(block_size
            .iter()
            .zip(&self.dims)
            .all(|(&block, &dim)| dim % block == 0 && block <= dim));

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

        let (sx, rx) = std::sync::mpsc::sync_channel(block_counts.iter().product());
        for_each_range_par_enumerate(block_counts_range.iter().cloned(), |i, block_idx| {
            let block_offset: Vec<_> = block_idx
                .iter()
                .zip(block_size)
                .map(|(&idx, &size)| idx * size)
                .collect();

            let mut block = VolumeBlock::new(block_size).unwrap();

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
            sx.send((i, block_decomp[0])).unwrap();

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

                    let rows = high.rows(0);
                    let mut stream = SerializeStream::new();
                    for row in rows {
                        for elem in row.as_slice().unwrap() {
                            elem.serialize(&mut stream);
                        }
                    }

                    let out_path = block_dir.join(format!("block_part_{counter}.bin"));
                    let out_file = File::create(out_path).unwrap();
                    stream.write_encode(out_file).unwrap();

                    counter += 1;
                }
            }
        });

        let mut superblock = VolumeBlock::new(&block_counts).unwrap();
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
            num_type: T::name().into(),
            dims: self.dims.clone(),
            block_size: block_size.into(),
            block_counts,
            input_block_dims: transformed.dims().into(),
            block_blueprints: BlockBlueprints::<T>::new(&block_resample_dims),
            filter,
        };
        output_header.serialize(&mut stream);
        for elem in transformed.flatten() {
            elem.serialize(&mut stream);
        }
        let output_path = output.as_ref().join("output.bin");
        let output_file = File::create(output_path).unwrap();
        stream.write_encode(output_file).unwrap();
    }

    unsafe fn flatten_idx_full_unchecked(&self, index: &[usize]) -> usize {
        flatten_idx_unchecked(&self.strides, &index[self.num_base_dims..])
    }
}

impl<'a, T: Num + Copy> Debug for VolumeWaveletEncoder<'a, T> {
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
    blueprints: BTreeMap<Vec<u32>, BlockBlueprint<T>>,
}

impl<T> BlockBlueprints<T> {
    pub fn new(dims: &[usize]) -> Self {
        let mut blueprints = BTreeMap::new();

        blueprints.insert(vec![0; dims.len()], BlockBlueprint::new(dims));

        Self {
            dims: dims.len(),
            blueprints,
        }
    }

    pub fn reconstruct_all(&self, block_path: impl AsRef<Path>, steps: &[u32]) -> VolumeBlock<T>
    where
        T: Deserializable + Num + Copy,
    {
        assert_eq!(self.dims, steps.len());

        for (k, b) in &self.blueprints {
            if k.iter().all(|&s| s == 0) {
                return b.reconstruct(block_path, steps);
            }
        }

        unreachable!()
    }

    #[allow(unused)]
    pub fn reconstruct(
        &self,
        block_path: impl AsRef<Path>,
        steps: &[u32],
        refinements: &[u32],
    ) -> VolumeBlock<T>
    where
        T: Deserializable + Num + Copy,
    {
        assert_eq!(self.dims, steps.len());
        assert_eq!(self.dims, refinements.len());

        let blueprint = self.blueprints.get(steps).unwrap();
        blueprint.reconstruct(block_path, refinements)
    }
}

impl<T: Serializable> Serializable for BlockBlueprints<T> {
    fn serialize(self, stream: &mut SerializeStream) {
        self.dims.serialize(stream);
        self.blueprints.serialize(stream);
    }
}

impl<T: Deserializable> Deserializable for BlockBlueprints<T> {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let dims = Deserializable::deserialize(stream);
        let blueprints = Deserializable::deserialize(stream);

        Self { dims, blueprints }
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
            for dim in 0..size.len() {
                if curr[dim] == 1 {
                    continue;
                }

                curr[dim] /= 2;
                let size = curr.clone();
                let mut offset = vec![0; curr.len()];
                offset[dim] = size[dim];

                parts.push(BlockBlueprintPart {
                    id,
                    dim,
                    size,
                    offset,
                });

                id += 1;
            }
        }

        Self {
            base_size,
            parts,
            _phantom: PhantomData,
        }
    }

    fn reconstruct(&self, block_path: impl AsRef<Path>, steps: &[u32]) -> VolumeBlock<T>
    where
        T: Deserializable + Num + Copy,
    {
        assert_eq!(steps.len(), self.base_size.len());

        let block_path = block_path.as_ref();
        let block_size: Vec<_> = steps
            .iter()
            .zip(&self.base_size)
            .map(|(&st, &si)| (si << st))
            .collect();
        let mut block = VolumeBlock::new(&block_size).unwrap();

        if steps.iter().all(|&s| s == 0) {
            return block;
        }

        let mut curr = vec![0; steps.len()];

        let mut block_window = block.window_mut();

        for part in &self.parts {
            if curr[part.dim] < steps[part.dim] {
                curr[part.dim] += 1;

                let block_path = block_path.join(format!("block_part_{}.bin", part.id));

                let f = std::fs::File::open(block_path).unwrap();
                let stream = DeserializeStream::new_decode(f).unwrap();
                let mut stream = stream.stream();

                let mut window = block_window.custom_window_mut(&part.offset, &part.size);
                let rows = window.rows_mut(0);
                for mut row in rows {
                    for elem in row.as_slice_mut().unwrap() {
                        *elem = Deserializable::deserialize(&mut stream);
                    }
                }

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
struct BlockBlueprintPart {
    id: usize,
    dim: usize,
    size: Vec<usize>,
    offset: Vec<usize>,
}

impl Serializable for BlockBlueprintPart {
    fn serialize(self, stream: &mut SerializeStream) {
        self.id.serialize(stream);
        self.dim.serialize(stream);
        self.size.serialize(stream);
        self.offset.serialize(stream);
    }
}

impl Deserializable for BlockBlueprintPart {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let id = Deserializable::deserialize(stream);
        let dim = Deserializable::deserialize(stream);
        let size = Deserializable::deserialize(stream);
        let offset = Deserializable::deserialize(stream);

        Self {
            id,
            dim,
            size,
            offset,
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
