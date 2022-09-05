use std::{borrow::Borrow, fs::File, ops::Range, path::Path};

use num_traits::Num;

use crate::{
    filter::Filter,
    stream::{AnyMap, Deserializable, Serializable, SerializeStream},
    transformations::{Chain, Lerp, Resample, Transformation, WaveletTransform},
    utilities::{flatten_idx, flatten_idx_unchecked},
    volume::VolumeBlock,
};

pub struct VolumeWaveletEncoder<'a, T: Num + Copy> {
    metadata: AnyMap,
    dims: Vec<usize>,
    num_base_dims: usize,
    fetchers: Vec<Option<VolumeFetcher<'a, T>>>,
}

pub(crate) struct OutputHeader<T> {
    pub num_type: String,
    pub metadata: AnyMap,
    pub dims: Vec<usize>,
    pub block_size: Vec<usize>,
    pub block_counts: Vec<usize>,
    pub block_resample_dims: Vec<usize>,
    pub resample_dims: Vec<usize>,
    pub input_block_dims: Vec<usize>,
    pub wavelet: T,
}

impl<T: Serializable> Serializable for OutputHeader<T> {
    fn serialize(self, stream: &mut SerializeStream) {
        T::name().serialize(stream);

        self.num_type.serialize(stream);
        self.metadata.serialize(stream);
        self.dims.serialize(stream);
        self.block_size.serialize(stream);
        self.block_counts.serialize(stream);
        self.block_resample_dims.serialize(stream);
        self.resample_dims.serialize(stream);
        self.input_block_dims.serialize(stream);
        self.wavelet.serialize(stream);
    }
}

impl<T: Deserializable> Deserializable for OutputHeader<T> {
    fn deserialize(stream: &mut crate::stream::DeserializeStream<'_>) -> Self {
        let t_name: String = Deserializable::deserialize(stream);
        assert_eq!(t_name, T::name());

        let num_type = Deserializable::deserialize(stream);
        let metadata = Deserializable::deserialize(stream);
        let dims = Deserializable::deserialize(stream);
        let block_size = Deserializable::deserialize(stream);
        let block_counts = Deserializable::deserialize(stream);
        let block_resample_dims = Deserializable::deserialize(stream);
        let resample_dims = Deserializable::deserialize(stream);
        let input_block_dims = Deserializable::deserialize(stream);
        let wavelet = Deserializable::deserialize(stream);

        Self {
            num_type,
            metadata,
            dims,
            block_size,
            block_counts,
            block_resample_dims,
            resample_dims,
            input_block_dims,
            wavelet,
        }
    }
}

pub(crate) struct BlockHeader {
    pub region: Vec<Range<usize>>,
}

impl Serializable for BlockHeader {
    fn serialize(self, stream: &mut SerializeStream) {
        self.region.serialize(stream)
    }
}

impl Deserializable for BlockHeader {
    fn deserialize(stream: &mut crate::stream::DeserializeStream<'_>) -> Self {
        let region = Deserializable::deserialize(stream);

        Self { region }
    }
}

type VolumeFetcher<'a, T> = Box<dyn FnMut(&[usize]) -> T + 'a>;

impl<'a, T: Serializable + Num + Lerp + Send + Copy> VolumeWaveletEncoder<'a, T> {
    pub fn new(dims: &[usize], num_base_dims: usize) -> Self {
        assert!(dims.len() >= num_base_dims);
        let num_fetchers = dims[num_base_dims..].iter().product();
        let mut fetchers = Vec::with_capacity(num_fetchers);
        for _ in 0..num_fetchers {
            fetchers.push(None);
        }

        Self {
            metadata: AnyMap::new(),
            dims: dims.into(),
            num_base_dims,
            fetchers,
        }
    }

    pub fn add_fetcher(&mut self, index: &[usize], f: impl FnMut(&[usize]) -> T + 'a) {
        let idx = flatten_idx(&self.dims[self.num_base_dims..], index);
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
        &mut self,
        output: impl AsRef<Path>,
        block_size: &[usize],
        wavelet: impl Filter<T> + Serializable + Clone,
    ) {
        assert_eq!(block_size.len(), self.dims.len());
        assert!(block_size
            .iter()
            .zip(&self.dims)
            .all(|(&block, &dim)| dim % block == 0));

        let block_resample_dims: Vec<_> = block_size
            .iter()
            .map(|size| size.next_power_of_two())
            .collect();
        let steps: Vec<_> = block_resample_dims
            .iter()
            .map(|size| size.ilog2())
            .collect();
        let resample = Resample::new(block_size, &block_resample_dims);
        let block_transform =
            Chain::from((resample, WaveletTransform::new(wavelet.clone(), &steps)));

        let block_counts: Vec<_> = block_size
            .iter()
            .zip(&self.dims)
            .map(|(&block, &dim)| dim / block)
            .collect();
        let mut block_idx = vec![0; block_size.len()];

        let num_elements = block_size.iter().product();
        let num_blocks = block_counts.iter().product();
        let mut superblock = VolumeBlock::new(&block_counts).unwrap();
        for i in 0..num_blocks {
            let block_offset: Vec<_> = block_idx
                .iter()
                .zip(block_size)
                .map(|(&idx, &size)| idx * size)
                .collect();

            let mut block = VolumeBlock::new(block_size).unwrap();
            let mut inner_idx = vec![0; block_size.len()];
            for i in 0..num_elements {
                let idx: Vec<_> = block_offset
                    .iter()
                    .zip(&inner_idx)
                    .map(|(&offset, &idx)| offset + idx)
                    .collect();
                let fetcher_idx = unsafe { self.flatten_idx_full_unchecked(&idx) };
                let fetcher = self.fetchers[fetcher_idx].as_mut().unwrap();
                block[i] = fetcher(&idx[0..self.num_base_dims]);

                for (idx, &size) in inner_idx.iter_mut().zip(block_size) {
                    *idx = (*idx + 1) % size;
                    if *idx != 0 {
                        break;
                    }
                }
            }

            let mut stream = SerializeStream::new();
            let transformed = block_transform.forwards(block);
            superblock[i] = transformed.flatten()[0];
            for elem in transformed.flatten().iter().skip(1) {
                elem.serialize(&mut stream);
            }

            let block_path = output.as_ref().join(format!("block_{i}.bin"));
            let mut block_file = File::create(block_path).unwrap();
            stream.write(&mut block_file).unwrap();

            for (block_idx, &count) in block_idx.iter_mut().zip(&block_counts) {
                *block_idx = (*block_idx + 1) % count;
                if *block_idx != 0 {
                    break;
                }
            }
        }

        let resample_dims: Vec<_> = block_counts
            .iter()
            .map(|size| size.next_power_of_two())
            .collect();
        let steps: Vec<_> = resample_dims.iter().map(|size| size.ilog2()).collect();
        let resample = Resample::new(&block_counts, &resample_dims);
        let superblock_transform =
            Chain::from((resample, WaveletTransform::new(wavelet.clone(), &steps)));
        let transformed = superblock_transform.forwards(superblock);

        let mut stream = SerializeStream::new();

        let superblock_header = OutputHeader {
            metadata: self.metadata.clone(),
            num_type: T::name().into(),
            dims: self.dims.clone(),
            block_size: block_size.into(),
            block_counts,
            block_resample_dims,
            resample_dims,
            input_block_dims: transformed.dims().into(),
            wavelet,
        };
        superblock_header.serialize(&mut stream);
        for elem in transformed.flatten() {
            elem.serialize(&mut stream);
        }
        let output_path = output.as_ref().join("output.bin");
        let mut output_file = File::create(output_path).unwrap();
        stream.write(&mut output_file).unwrap();
    }

    unsafe fn flatten_idx_full_unchecked(&self, index: &[usize]) -> usize {
        flatten_idx_unchecked(
            &self.dims[self.num_base_dims..],
            &index[self.num_base_dims..],
        )
    }
}

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use crate::{filter::AverageFilter, volume::VolumeBlock};

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
}
