use std::{borrow::Borrow, fs::File, ops::Range, path::Path};

use num_traits::Num;

use crate::{
    filter::Filter,
    range::for_each_range_enumerate,
    stream::{AnyMap, Deserializable, Serializable, SerializeStream},
    transformations::{
        Chain, Lerp, ResampleCfg, ResampleExtend, ReversibleTransform, WaveletDecompCfg,
        WaveletTransform,
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

pub(crate) struct OutputHeader<T> {
    pub num_type: String,
    pub metadata: AnyMap,
    pub dims: Vec<usize>,
    pub block_size: Vec<usize>,
    pub block_counts: Vec<usize>,
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
        self.input_block_dims.serialize(stream);
        self.wavelet.serialize(stream);
    }
}

impl<T: Deserializable> Deserializable for OutputHeader<T> {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let t_name: String = Deserializable::deserialize(stream);
        assert_eq!(t_name, T::name());

        let num_type = Deserializable::deserialize(stream);
        let metadata = Deserializable::deserialize(stream);
        let dims = Deserializable::deserialize(stream);
        let block_size = Deserializable::deserialize(stream);
        let block_counts = Deserializable::deserialize(stream);
        let input_block_dims = Deserializable::deserialize(stream);
        let wavelet = Deserializable::deserialize(stream);

        Self {
            num_type,
            metadata,
            dims,
            block_size,
            block_counts,
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
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let region = Deserializable::deserialize(stream);

        Self { region }
    }
}

type VolumeFetcher<'a, T> = Box<dyn FnMut(&[usize]) -> T + 'a>;

impl<'a, T: Serializable + Num + Lerp + Send + Copy> VolumeWaveletEncoder<'a, T> {
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

    pub fn add_fetcher(&mut self, index: &[usize], f: impl FnMut(&[usize]) -> T + 'a) {
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
        &mut self,
        output: impl AsRef<Path>,
        block_size: &[usize],
        wavelet: impl Filter<T> + Serializable + Clone,
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
        let block_transform = Chain::from((
            ResampleExtend,
            WaveletTransform::new(wavelet.clone(), false),
        ));

        let block_counts: Vec<_> = block_size
            .iter()
            .zip(&self.dims)
            .map(|(&block, &dim)| dim / block)
            .collect();
        let block_counts_range: Vec<_> = block_counts.iter().map(|&c| 0..c).collect();
        let block_range: Vec<_> = block_size.iter().map(|&b| 0..b).collect();

        let mut superblock = VolumeBlock::new(&block_counts).unwrap();
        for_each_range_enumerate(block_counts_range.iter().cloned(), |i, block_idx| {
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
                let fetcher = self.fetchers[fetcher_idx].as_mut().unwrap();
                block[i] = fetcher(&idx[0..self.num_base_dims]);
            });

            let block_decomp = block_transform.forwards(block, block_transform_cfg);
            superblock[i] = block_decomp[0];

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

        let resample_dims: Vec<_> = block_counts
            .iter()
            .map(|size| size.next_power_of_two())
            .collect();
        let steps: Vec<_> = resample_dims.iter().map(|size| size.ilog2()).collect();

        let output_transform_cfg = Chain::from((
            ResampleCfg::new(&resample_dims),
            WaveletDecompCfg::new(&steps),
        ));
        let output_transform = Chain::from((
            ResampleExtend,
            WaveletTransform::new(wavelet.clone(), false),
        ));
        let transformed = output_transform.forwards(superblock, output_transform_cfg);

        let mut stream = SerializeStream::new();

        let output_header = OutputHeader {
            metadata: self.metadata.clone(),
            num_type: T::name().into(),
            dims: self.dims.clone(),
            block_size: block_size.into(),
            block_counts,
            input_block_dims: transformed.dims().into(),
            wavelet,
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
