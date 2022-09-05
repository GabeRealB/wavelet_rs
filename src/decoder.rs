use std::{
    borrow::Borrow,
    ops::Range,
    path::{Path, PathBuf},
};

use num_traits::Num;

use crate::{
    encoder::OutputHeader,
    filter::Filter,
    stream::{AnyMap, Deserializable, DeserializeStream},
    transformations::{Chain, Lerp, Resample, Transformation, WaveletTransform},
    volume::VolumeBlock,
};

pub struct VolumeWaveletDecoder<
    T: Deserializable + Num + Lerp + Send + Copy,
    F: Filter<T> + Deserializable + Clone,
> {
    path: PathBuf,
    metadata: AnyMap,
    dims: Vec<usize>,
    block_size: Vec<usize>,
    block_counts: Vec<usize>,
    block_resample_dims: Vec<usize>,
    resample_dims: Vec<usize>,
    input_block_dims: Vec<usize>,
    wavelet: F,
    input_block: VolumeBlock<T>,
}

impl<T: Deserializable + Num + Lerp + Send + Copy, F: Filter<T> + Deserializable + Clone>
    VolumeWaveletDecoder<T, F>
{
    pub fn new(p: impl AsRef<Path>) -> Self {
        let p = p.as_ref();
        let input_path = p.parent().unwrap().to_path_buf();

        let input = std::fs::read(p).unwrap();
        let mut stream = DeserializeStream::new(&input);
        let header: OutputHeader<_> = Deserializable::deserialize(&mut stream);
        let num_elements = header.input_block_dims.iter().product();
        let mut elements = Vec::with_capacity(num_elements);
        for _ in 0..num_elements {
            elements.push(T::deserialize(&mut stream));
        }
        let input_block = VolumeBlock::new_with_data(&header.input_block_dims, elements).unwrap();

        assert_eq!(header.dims.len(), header.block_size.len());
        assert_eq!(header.block_counts.len(), header.block_size.len());
        assert_eq!(header.resample_dims.len(), header.block_size.len());
        assert_eq!(header.input_block_dims.len(), header.block_size.len());

        Self {
            path: input_path,
            metadata: header.metadata,
            dims: header.dims,
            block_size: header.block_size,
            block_counts: header.block_counts,
            block_resample_dims: header.block_resample_dims,
            resample_dims: header.resample_dims,
            input_block_dims: header.input_block_dims,
            wavelet: header.wavelet,
            input_block,
        }
    }

    pub fn get_metadata<Q, M>(&self, key: &Q) -> Option<M>
    where
        String: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
        M: Deserializable,
    {
        self.metadata.get(key)
    }

    pub fn decode(
        &self,
        mut writer: impl FnMut(&[usize], T),
        roi: &[Range<usize>],
        levels: &[u32],
    ) {
        assert_eq!(roi.len(), self.dims.len());
        assert_eq!(levels.len(), self.dims.len());

        assert!(roi
            .iter()
            .zip(&self.dims)
            .all(|(region, volume_size)| !region.contains(volume_size)));

        let input_steps: Vec<_> = self
            .input_block_dims
            .iter()
            .zip(levels)
            .map(|(&dim, &level)| dim.ilog2().min(level))
            .collect();
        let block_steps: Vec<_> = input_steps
            .iter()
            .zip(levels)
            .map(|(&input_step, &level)| level - input_step)
            .collect();

        let resample = Resample::new(&self.block_counts, &self.resample_dims);
        let input_transform = Chain::from((
            resample,
            WaveletTransform::new(self.wavelet.clone(), &input_steps),
        ));
        let first_pass = input_transform.backwards(self.input_block.clone());

        let resample = Resample::new(&self.block_size, &self.block_resample_dims);
        let block_transform = Chain::from((
            resample,
            WaveletTransform::new(self.wavelet.clone(), &block_steps),
        ));

        let num_blocks = self.block_counts.iter().product();
        let mut block_iter_idx = vec![0; self.block_size.len()];

        for block_idx in 0..num_blocks {
            let block_offset: Vec<_> = block_iter_idx
                .iter()
                .zip(&self.block_size)
                .map(|(&idx, &size)| idx * size)
                .collect();
            let block_range: Vec<_> = block_offset
                .iter()
                .zip(&self.block_size)
                .map(|(&start, &size)| start..start + size)
                .collect();

            if roi.iter().zip(&block_range).all(|(required, range)| {
                required.contains(&range.start) || required.contains(&range.end)
            }) {
                let block_path = self.path.join(format!("block_{block_idx}.bin"));
                let block_buffer = std::fs::read(block_path).unwrap();
                let mut stream = DeserializeStream::new(&block_buffer);

                let mut block = VolumeBlock::new(&self.block_size).unwrap();
                block.flatten_mut()[0] = first_pass.flatten()[0];
                for elem in block.flatten_mut().iter_mut().skip(1) {
                    *elem = Deserializable::deserialize(&mut stream);
                }

                let block_pass = block_transform.backwards(block);

                let sub_range: Vec<_> = roi
                    .iter()
                    .zip(block_range)
                    .map(|(required, range)| {
                        required.start.max(range.start)..required.end.min(range.end)
                    })
                    .collect();
                let num_range_elems = sub_range.iter().map(|r| r.end - r.start).product();
                let mut elem_idx: Vec<_> = sub_range.iter().map(|range| range.start).collect();

                for _ in 0..num_range_elems {
                    let local_idx: Vec<_> = elem_idx
                        .iter()
                        .zip(&block_offset)
                        .map(|(&global, &offset)| global - offset)
                        .collect();

                    let elem = block_pass[&*local_idx];
                    writer(&elem_idx, elem);

                    for (elem_idx, range) in elem_idx.iter_mut().zip(&sub_range) {
                        *elem_idx += 1;
                        if *elem_idx == range.end {
                            *elem_idx = range.start;
                        }

                        if *elem_idx != range.start {
                            break;
                        }
                    }
                }
            }

            for (block_idx, &count) in block_iter_idx.iter_mut().zip(&self.block_counts) {
                *block_idx = (*block_idx + 1) % count;
                if *block_idx != 0 {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use crate::{decoder::VolumeWaveletDecoder, filter::AverageFilter, volume::VolumeBlock};

    const TRANSFORM_ERROR: f32 = 0.001;

    #[test]
    fn decode() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test/decode/output.bin");

        let decoder = VolumeWaveletDecoder::<f32, AverageFilter>::new(res_path);

        let dims = [256, 256, 256];
        let num_elements = dims.iter().product();
        let mut block_1 = VolumeBlock::new(&dims).unwrap();
        let mut block_2 = VolumeBlock::new(&dims).unwrap();
        let writer = |idx: &[usize], elem: f32| {
            let block_idx = idx[3];
            let idx = &idx[..3];

            if block_idx == 0 {
                block_1[idx] = elem
            } else if block_idx == 1 {
                block_2[idx] = elem
            }
        };

        let volume_dims = [256usize, 256usize, 256usize, 2usize];
        let roi = volume_dims.map(|dim| 0..dim);
        let steps = volume_dims.map(|dim| dim.ilog2());
        decoder.decode(writer, &roi, &steps);

        let expected_block_1 =
            VolumeBlock::new_with_data(&dims, vec![1.0f32; num_elements]).unwrap();
        let expected_block_2 =
            VolumeBlock::new_with_data(&dims, vec![2.0f32; num_elements]).unwrap();

        assert!(block_1.is_equal(&expected_block_1, TRANSFORM_ERROR));
        assert!(block_2.is_equal(&expected_block_2, TRANSFORM_ERROR));
    }
}
