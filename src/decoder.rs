use std::{
    borrow::Borrow,
    ops::Range,
    path::{Path, PathBuf},
};

use num_traits::Num;

use crate::{
    encoder::{BlockBlueprints, OutputHeader},
    filter::Filter,
    range::{for_each_range, for_each_range_enumerate},
    stream::{AnyMap, Deserializable, DeserializeStream},
    transformations::{
        Chain, ResampleCfg, ResampleExtend, ResampleIScale, Reverse, ReversibleTransform,
        WaveletRecompCfg, WaveletTransform,
    },
    volume::VolumeBlock,
};

#[derive(Debug)]
pub struct VolumeWaveletDecoder<
    T: Deserializable + Num + Send + Copy,
    F: Filter<T> + Deserializable + Clone,
> {
    path: PathBuf,
    metadata: AnyMap,
    dims: Vec<usize>,
    block_size: Vec<usize>,
    block_counts: Vec<usize>,
    input_block_dims: Vec<usize>,
    input_block: VolumeBlock<T>,
    block_blueprints: BlockBlueprints<T>,
    filter: F,
}

impl<T: Deserializable + Num + Send + Copy, F: Filter<T> + Deserializable + Clone>
    VolumeWaveletDecoder<T, F>
{
    pub fn new(p: impl AsRef<Path>) -> Self {
        let p = p.as_ref();
        let input_path = p.parent().unwrap().to_path_buf();

        let f = std::fs::File::open(p).unwrap();
        let stream = DeserializeStream::new_decode(f).unwrap();
        let mut stream = stream.stream();
        let header: OutputHeader<_, _> = Deserializable::deserialize(&mut stream);
        let num_elements = header.input_block_dims.iter().product();
        let mut elements = Vec::with_capacity(num_elements);
        for _ in 0..num_elements {
            elements.push(T::deserialize(&mut stream));
        }
        let input_block = VolumeBlock::new_with_data(&header.input_block_dims, elements).unwrap();

        assert_eq!(header.dims.len(), header.block_size.len());
        assert_eq!(header.block_counts.len(), header.block_size.len());
        assert_eq!(header.input_block_dims.len(), header.block_size.len());

        Self {
            path: input_path,
            metadata: header.metadata,
            dims: header.dims,
            block_size: header.block_size,
            block_counts: header.block_counts,
            input_block_dims: header.input_block_dims,
            block_blueprints: header.block_blueprints,
            filter: header.filter,
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

    pub fn refine(
        &self,
        mut reader: impl FnMut(&[usize]) -> T,
        mut writer: impl FnMut(&[usize], T),
        input_range: &[Range<usize>],
        output_range: &[Range<usize>],
        curr_levels: &[u32],
        refinements: &[u32],
    ) {
        assert_eq!(input_range.len(), self.dims.len());
        assert_eq!(output_range.len(), self.dims.len());
        assert_eq!(curr_levels.len(), self.dims.len());
        assert_eq!(refinements.len(), self.dims.len());

        // Input must be able to populate all the necessary blocks.
        assert!(input_range
            .iter()
            .zip(&self.dims)
            .zip(&self.block_size)
            .all(
                |((region, &volume_size), &block)| (region.end <= volume_size)
                    && (region.start % block == 0)
                    && (region.end % block == 0)
            ));

        // Output range must be contained inside the received input range.
        assert!(input_range
            .iter()
            .zip(output_range)
            .all(|(i_range, o_range)| (i_range.start <= o_range.start)
                && (o_range.end <= i_range.end)));

        let input_levels: Vec<_> = curr_levels
            .iter()
            .zip(&self.input_block_dims)
            .map(|(&lvl, &i_dim)| lvl.min(i_dim.ilog2()))
            .collect();
        let block_levels: Vec<_> = curr_levels
            .iter()
            .zip(&input_levels)
            .map(|(&lvl, &i_lvl)| lvl - i_lvl)
            .collect();

        let input_refinements: Vec<_> = input_levels
            .iter()
            .zip(refinements)
            .zip(&self.input_block_dims)
            .map(|((&i_lvl, &re), &i_dim)| (i_lvl + re).min(i_dim.ilog2()) - i_lvl)
            .collect();
        let block_refinements: Vec<_> = input_refinements
            .iter()
            .zip(refinements)
            .map(|(&i_re, &re)| re - i_re)
            .collect();

        let requires_input_pass = input_refinements.iter().any(|&r| r != 0);
        if requires_input_pass {
            let new_levels: Vec<_> = curr_levels
                .iter()
                .zip(refinements)
                .map(|(&c, &r)| c + r)
                .collect();
            return self.decode(writer, output_range, &new_levels);
        }

        let resample_dim: Vec<_> = self
            .block_size
            .iter()
            .map(|&b| b.next_power_of_two())
            .collect();
        let block_decompositions: Vec<_> = resample_dim.iter().map(|&b| b.ilog2()).collect();
        let remaining_decompositions: Vec<_> = block_decompositions
            .iter()
            .zip(&block_levels)
            .map(|(&d, &l)| d - l)
            .collect();

        /* let block_forwards_steps: Vec<_> = self
        .block_blueprints
        .block_size(&block_levels, &block_refinements)
        .into_iter()
        .map(|s| s.ilog2())
        .collect(); */

        let block_input_dims: Vec<_> = block_levels.iter().map(|&re| 2usize.pow(re)).collect();
        let block_downsample_stepping: Vec<_> = resample_dim
            .iter()
            .zip(&block_input_dims)
            .map(|(&s, &d)| s / d)
            .collect();

        let resample_cfg = ResampleCfg::new(&resample_dim);
        let resample_pass = ResampleExtend;

        let refinement_cfg = Chain::combine(
            ResampleCfg::new(&self.block_size),
            ResampleCfg::new(&resample_dim),
        )
        .chain(WaveletRecompCfg::new_with_start_dim(
            &block_refinements,
            &block_refinements,
            self.block_blueprints
                .start_dim(&remaining_decompositions, &block_refinements),
        ));
        let refinement_pass = Chain::combine(ResampleExtend, Reverse::new(ResampleIScale))
            .chain(WaveletTransform::new(self.filter.clone(), false));

        /* let block_refinement_info =
        RefinementInfo::new(&resample_dim, &block_decompositions, &block_levels); */

        let num_blocks = self.block_counts.iter().product();
        let mut block_iter_idx = vec![0; self.block_size.len()];

        let num_elements = self.block_size.iter().product();
        let num_elements_downs = block_input_dims.iter().product();

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

            if input_range
                .iter()
                .zip(block_range)
                .all(|(i_r, b_r)| (i_r.start <= b_r.start) && (i_r.end >= b_r.end))
            {
                let mut block_input = VolumeBlock::new(&self.block_size).unwrap();
                let mut elem_idx = vec![0; self.block_size.len()];
                for _ in 0..num_elements {
                    let idx: Vec<_> = block_offset
                        .iter()
                        .zip(&elem_idx)
                        .map(|(&offset, &idx)| offset + idx)
                        .collect();
                    block_input[&*elem_idx] = reader(&idx);

                    for (idx, &size) in elem_idx.iter_mut().zip(&self.block_size) {
                        *idx = (*idx + 1) % size;
                        if *idx != 0 {
                            break;
                        }
                    }
                }

                let block_input = resample_pass.forwards(block_input, resample_cfg);

                let block_path = self.path.join(format!("block_{block_idx}"));
                let mut block = self.block_blueprints.reconstruct(
                    &self.filter,
                    block_path,
                    &block_levels,
                    &block_refinements,
                );

                let mut block_window = block.window_mut();
                let mut block_window = block_window.custom_range_mut(&block_input_dims);

                let mut elem_idx = vec![0; self.block_size.len()];
                for _ in 0..num_elements_downs {
                    let idx: Vec<_> = block_downsample_stepping
                        .iter()
                        .zip(&elem_idx)
                        .map(|(&step, &idx)| step * idx)
                        .collect();
                    block_window[&*elem_idx] = block_input[&*idx];

                    for (idx, &size) in elem_idx.iter_mut().zip(&block_input_dims) {
                        *idx = (*idx + 1) % size;
                        if *idx != 0 {
                            break;
                        }
                    }
                }

                let block = refinement_pass.backwards(block, refinement_cfg);
                let mut elem_idx = vec![0; self.block_size.len()];
                for _ in 0..num_elements {
                    let idx: Vec<_> = block_offset
                        .iter()
                        .zip(&elem_idx)
                        .map(|(&offset, &idx)| offset + idx)
                        .collect();
                    writer(&idx, block[&*elem_idx]);

                    for (idx, &size) in elem_idx.iter_mut().zip(&self.block_size) {
                        *idx = (*idx + 1) % size;
                        if *idx != 0 {
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

        let input_backwards_steps: Vec<_> = self
            .input_block_dims
            .iter()
            .zip(levels)
            .map(|(&dim, &level)| dim.ilog2().min(level))
            .collect();
        let block_backwards_steps: Vec<_> = input_backwards_steps
            .iter()
            .zip(levels)
            .map(|(&input_step, &level)| level - input_step)
            .collect();

        let input_forwards_steps: Vec<_> = self
            .input_block_dims
            .iter()
            .map(|&dim| dim.ilog2())
            .collect();
        let block_forwards_steps: Vec<_> = self
            .block_blueprints
            .block_size_full(&block_backwards_steps)
            .into_iter()
            .map(|s| s.ilog2())
            .collect();

        let input_transform_cfg = Chain::combine(
            ResampleCfg::new(&self.block_counts),
            WaveletRecompCfg::new(&input_forwards_steps, &input_backwards_steps),
        );
        let input_transform = Chain::combine(
            ResampleExtend,
            WaveletTransform::new(self.filter.clone(), false),
        );
        let first_pass = input_transform.backwards(self.input_block.clone(), input_transform_cfg);

        let block_transform_cfg = Chain::combine(
            ResampleCfg::new(&self.block_size),
            ResampleCfg::new(&self.block_size),
        )
        .chain(WaveletRecompCfg::new_with_start_dim(
            &block_forwards_steps,
            &block_backwards_steps,
            self.block_blueprints.start_dim_full(&block_backwards_steps),
        ));
        let block_transform = Chain::combine(ResampleExtend, Reverse::new(ResampleIScale))
            .chain(WaveletTransform::new(self.filter.clone(), false));

        let block_range: Vec<_> = self.block_counts.iter().map(|&c| 0..c).collect();
        for_each_range_enumerate(block_range.iter().cloned(), |block_idx, block_iter_idx| {
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
                let block_path = self.path.join(format!("block_{block_idx}"));
                let mut block = self.block_blueprints.reconstruct_full(
                    &self.filter,
                    block_path,
                    &block_backwards_steps,
                );
                block[0] = first_pass[block_idx];

                let block_pass = block_transform.backwards(block, block_transform_cfg);

                let sub_range: Vec<_> = roi
                    .iter()
                    .zip(block_range)
                    .map(|(required, range)| {
                        required.start.max(range.start)..required.end.min(range.end)
                    })
                    .collect();
                for_each_range(sub_range.iter().cloned(), |idx| {
                    let local_idx: Vec<_> = idx
                        .iter()
                        .zip(&block_offset)
                        .map(|(&global, &offset)| global - offset)
                        .collect();

                    let elem = block_pass[&*local_idx];
                    writer(idx, elem);
                });
            }
        });
    }
}

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use crate::{
        decoder::VolumeWaveletDecoder, filter::AverageFilter, vector::Vector, volume::VolumeBlock,
    };

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

    #[test]
    fn decode_img_1() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test/decode_img_1/");

        let data_path = res_path.join("output.bin");

        let decoder = VolumeWaveletDecoder::<Vector<f32, 3>, AverageFilter>::new(data_path);

        let dims = [2048, 2048, 1];
        let range = dims.map(|d| 0..d);
        let steps = 2048usize.ilog2();
        let mut data = VolumeBlock::new(&dims).unwrap();

        for x in 0..steps + 1 {
            let writer = |idx: &[usize], elem| {
                data[idx] = elem;
            };

            decoder.decode(writer, &range, &[x, steps, 0]);
            let mut img = image::Rgb32FImage::new(data.dims()[0] as u32, data.dims()[1] as u32);
            for (p, rgb) in img.pixels_mut().zip(data.flatten()) {
                p.0 = *rgb.as_ref();
            }
            let img = image::DynamicImage::ImageRgb32F(img).into_rgb8();
            img.save(res_path.join(format!("img_1_x_{x}.png"))).unwrap();
        }
    }

    #[test]
    fn decode_img_1_refine() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test/decode_img_1/");

        let data_path = res_path.join("output.bin");

        let decoder = VolumeWaveletDecoder::<Vector<f32, 3>, AverageFilter>::new(data_path);

        let dims = [2048, 2048, 1];
        let range = dims.map(|d| 0..d);
        let steps = 2048usize.ilog2();
        let mut data = VolumeBlock::new(&dims).unwrap();
        let writer = |idx: &[usize], elem| {
            data[idx] = elem;
        };

        decoder.decode(writer, &range, &[0, steps, 0]);
        let mut img = image::Rgb32FImage::new(data.dims()[0] as u32, data.dims()[1] as u32);
        for (p, rgb) in img.pixels_mut().zip(data.flatten()) {
            p.0 = *rgb.as_ref();
        }
        let img = image::DynamicImage::ImageRgb32F(img).into_rgb8();
        img.save(res_path.join("img_1_ref_x_0.png")).unwrap();

        for x in 1..steps + 1 {
            let mut next = VolumeBlock::new(&dims).unwrap();
            let reader = |idx: &[usize]| data[idx];
            let writer = |idx: &[usize], elem| {
                next[idx] = elem;
            };

            decoder.refine(
                reader,
                writer,
                &range,
                &range,
                &[x - 1, steps, 0],
                &[1, 0, 0],
            );
            data = next;

            let mut img = image::Rgb32FImage::new(data.dims()[0] as u32, data.dims()[1] as u32);
            for (p, rgb) in img.pixels_mut().zip(data.flatten()) {
                p.0 = *rgb.as_ref();
            }
            let img = image::DynamicImage::ImageRgb32F(img).into_rgb8();
            img.save(res_path.join(format!("img_1_ref_x_{x}.png")))
                .unwrap();
        }
    }
}
