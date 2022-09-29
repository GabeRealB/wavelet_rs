//! Utilities for decoding a processed dataset.

use num_traits::Zero;
use std::{
    borrow::Borrow,
    ops::Range,
    path::{Path, PathBuf},
};

use crate::{
    encoder::{BlockBlueprints, OutputHeader},
    filter::Filter,
    range::{for_each_range, for_each_range_enumerate, for_each_range_par_enumerate},
    stream::{AnyMap, Deserializable, DeserializeStream},
    transformations::{
        Chain, ResampleCfg, ResampleExtend, ResampleIScale, Reverse, ReversibleTransform,
        WaveletRecompCfg, WaveletTransform,
    },
    volume::VolumeBlock,
};

/// Decoder for a dataset encoded with a wavelet transform.
#[derive(Debug)]
pub struct VolumeWaveletDecoder<T, F> {
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

impl<T, F> VolumeWaveletDecoder<T, F>
where
    T: Zero + Deserializable + Clone + Send + Sync,
    F: Filter<T> + Deserializable + Clone,
{
    /// Constructs a new decoder.
    ///
    /// The provided path must point to the root of the
    /// dataset, usually a file named `output.bin`.
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

    /// Fetches the value associated with a key from the dataset.
    pub fn get_metadata<Q, M>(&self, key: &Q) -> Option<M>
    where
        String: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
        M: Deserializable,
    {
        self.metadata.get(key)
    }

    /// Applies a partial decoding to a partially decoded dataset.
    pub fn refine<BR, R, BW, W>(
        &self,
        reader_fetcher: impl FnOnce(&[usize]) -> BR,
        writer_fetcher: impl FnOnce(&[usize]) -> BW,
        input_range: &[Range<usize>],
        output_range: &[Range<usize>],
        curr_levels: &[u32],
        refinements: &[u32],
    ) where
        BR: Fn(usize) -> R + Sync,
        R: Fn(&[usize]) -> T,
        BW: Fn(usize) -> W + Sync,
        W: FnMut(&[usize], T),
    {
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
            return self.decode(writer_fetcher, output_range, &new_levels);
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

        let block_input_dims: Vec<_> = block_levels.iter().map(|&re| 2usize.pow(re)).collect();
        let block_downsample_stepping: Vec<_> = resample_dim
            .iter()
            .zip(&block_input_dims)
            .map(|(&s, &d)| s / d)
            .collect();

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

        let block_range: Vec<_> = self.block_counts.iter().map(|&c| 0..c).collect();
        let block_input_range: Vec<_> = block_input_dims.iter().map(|&c| 0..c).collect();
        let block_size_range: Vec<_> = self.block_size.iter().map(|&c| 0..c).collect();

        let readers = reader_fetcher(&self.block_counts);
        let writers = writer_fetcher(&self.block_counts);

        for_each_range_par_enumerate(block_range.iter().cloned(), |block_idx, block_iter_idx| {
            let reader = readers(block_idx);
            let mut writer = writers(block_idx);

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
                let block_path = self.path.join(format!("block_{block_idx}"));
                let mut block = self.block_blueprints.reconstruct(
                    &self.filter,
                    block_path,
                    &block_levels,
                    &block_refinements,
                );

                // Load the partially recomposed data into the block.
                for_each_range(block_input_range.iter().cloned(), |local_idx| {
                    let reader_idx: Vec<_> = block_downsample_stepping
                        .iter()
                        .zip(local_idx)
                        .map(|(&step, &idx)| step * idx)
                        .collect();

                    block[local_idx] = reader(&reader_idx);
                });

                let block = refinement_pass.backwards(block, refinement_cfg);

                // Copy the result back to the caller.
                for_each_range_enumerate(block_size_range.iter().cloned(), |i, local_idx| {
                    writer(local_idx, block[i].clone());
                });
            }
        });
    }

    /// Decodes the dataset.
    pub fn decode<BW, W>(
        &self,
        writer_fetcher: impl FnOnce(&[usize]) -> BW,
        roi: &[Range<usize>],
        levels: &[u32],
    ) where
        BW: Fn(usize) -> W + Sync,
        W: FnMut(&[usize], T),
    {
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
        let block_forwards_steps = self
            .block_blueprints
            .block_decompositions_full(&block_backwards_steps);

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

        let writers = writer_fetcher(&self.block_counts);

        // The dataset was encoded using a two pass approach.
        // The first pass deconstructs each block independently. Since each
        // decomposition step halves the size of the decomposed axis, it would
        // only allow for a partial decomposition, if the used block size is
        // smaller than the size of the dataset. To mitigate this, we collect
        // the computed approximation coefficients into a new block, and decompose
        // it in a second pass. To recompose the original dataset, we simply
        // follow the same procedure backwards.
        let input_transform_cfg = Chain::combine(
            ResampleCfg::new(&self.block_counts),
            WaveletRecompCfg::new(&input_forwards_steps, &input_backwards_steps),
        );
        let input_transform = Chain::combine(
            ResampleExtend,
            WaveletTransform::new(self.filter.clone(), false),
        );
        let first_pass = input_transform.backwards(self.input_block.clone(), input_transform_cfg);

        // Iterate over each block and decode it, if it is part of the requested data range.
        let block_range: Vec<_> = self.block_counts.iter().map(|&c| 0..c).collect();
        for_each_range_par_enumerate(block_range.iter().cloned(), |block_idx, block_iter_idx| {
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

            // Check whether the block contains data in the requested range.
            if roi.iter().zip(&block_range).all(|(required, range)| {
                required.contains(&range.start) || required.contains(&range.end)
            }) {
                // To recompose a block we need to reconstruct the block of coefficients.
                // The detail coefficients are stored on disk, while the approximation coefficient
                // (i.e. the first element) is contained in the block from the first reconstruction
                // pass.
                let block_path = self.path.join(format!("block_{block_idx}"));
                let mut block = self.block_blueprints.reconstruct_full(
                    &self.filter,
                    block_path,
                    &block_backwards_steps,
                );
                block[0] = first_pass[block_idx].clone();

                let block_pass = block_transform.backwards(block, block_transform_cfg);

                // Fetch the writer for the current block.
                let mut writer = writers(block_idx);

                // Write back all elements of required range, which are contained in the reconstructed block.
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

                    let elem = block_pass[&*local_idx].clone();
                    writer(&local_idx, elem);
                });
            }
        });
    }
}

#[cfg(test)]
mod test {
    use std::{path::PathBuf, sync::Mutex};

    use crate::{
        decoder::VolumeWaveletDecoder, filter::AverageFilter, vector::Vector, volume::VolumeBlock,
    };

    const TRANSFORM_ERROR: f32 = 0.001;

    struct ForceOnce;

    impl ForceOnce {
        fn consume(self) {}
    }

    #[test]
    fn decode() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test/decode/output.bin");

        let decoder = VolumeWaveletDecoder::<f32, AverageFilter>::new(res_path);

        let dims = [256, 256, 256];
        let num_elements = dims.iter().product();
        let mut block_1 = VolumeBlock::new_zero(&dims).unwrap();
        let mut block_2 = VolumeBlock::new_zero(&dims).unwrap();

        let force_once = ForceOnce;
        let writer = |counts: &[usize]| {
            force_once.consume();

            let block_1_windows = block_1.window_mut().divide_into_mut(counts);
            let block_2_windows = block_2.window_mut().divide_into_mut(counts);

            let (block_1_windows, _, _) = block_1_windows.into_raw_parts();
            let (block_2_windows, _, _) = block_2_windows.into_raw_parts();

            let windows: Vec<_> = block_1_windows
                .into_iter()
                .zip(block_2_windows)
                .map(|a| Mutex::new(Some(a)))
                .collect();

            move |block_idx: usize| {
                let window = windows.get(block_idx).unwrap();
                let (mut block_1, mut block_2) = window.lock().unwrap().take().unwrap();

                move |idx: &[usize], elem: f32| {
                    let block_idx = idx[3];
                    let idx = &idx[..3];

                    if block_idx == 0 {
                        block_1[idx] = elem
                    } else if block_idx == 1 {
                        block_2[idx] = elem
                    }
                }
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
    fn decode_sample() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test/decode_sample/");

        let data_path = res_path.join("output.bin");
        let decoder = VolumeWaveletDecoder::<f32, AverageFilter>::new(data_path);

        let dims = [4, 4, 1];
        let range = dims.map(|d| 0..d);
        let steps = 4usize.ilog2();
        let mut data = VolumeBlock::new_zero(&dims).unwrap();

        let force_once = ForceOnce;
        let writer = |counts: &[usize]| {
            force_once.consume();

            let windows = data.window_mut().divide_into_mut(counts);

            let (windows, _, _) = windows.into_raw_parts();
            let windows: Vec<_> = windows.into_iter().map(|w| Mutex::new(Some(w))).collect();

            move |block_idx: usize| {
                let window = &windows[block_idx];
                let mut window = window.lock().unwrap().take().unwrap();

                move |idx: &[usize], elem| {
                    window[idx] = elem;
                }
            }
        };

        decoder.decode(writer, &range, &[0, steps, 0]);

        for x in 1..steps + 1 {
            let mut next = VolumeBlock::new_zero(&dims).unwrap();
            let reader = |counts: &[usize]| {
                let windows = data.window().divide_into(counts);

                let (windows, _, _) = windows.into_raw_parts();
                let windows: Vec<_> = windows.into_iter().map(|w| Mutex::new(Some(w))).collect();

                move |block_idx: usize| {
                    let window = &windows[block_idx];
                    let window = window.lock().unwrap().take().unwrap();
                    move |idx: &[usize]| window[idx]
                }
            };

            let force_once = ForceOnce;
            let writer = |counts: &[usize]| {
                force_once.consume();

                let windows = next.window_mut().divide_into_mut(counts);

                let (windows, _, _) = windows.into_raw_parts();
                let windows: Vec<_> = windows.into_iter().map(|w| Mutex::new(Some(w))).collect();

                move |block_idx: usize| {
                    let window = &windows[block_idx];
                    let mut window = window.lock().unwrap().take().unwrap();

                    move |idx: &[usize], elem| {
                        window[idx] = elem;
                    }
                }
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
        }
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
        let mut data = VolumeBlock::new_zero(&dims).unwrap();

        for x in 0..steps + 1 {
            let force_once = ForceOnce;
            let writer = |counts: &[usize]| {
                force_once.consume();

                let windows = data.window_mut().divide_into_mut(counts);

                let (windows, _, _) = windows.into_raw_parts();
                let windows: Vec<_> = windows.into_iter().map(|w| Mutex::new(Some(w))).collect();

                move |block_idx: usize| {
                    let window = &windows[block_idx];
                    let mut window = window.lock().unwrap().take().unwrap();

                    move |idx: &[usize], elem| {
                        window[idx] = elem;
                    }
                }
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
        let mut data = VolumeBlock::new_zero(&dims).unwrap();

        let force_once = ForceOnce;
        let writer = |counts: &[usize]| {
            force_once.consume();

            let windows = data.window_mut().divide_into_mut(counts);

            let (windows, _, _) = windows.into_raw_parts();
            let windows: Vec<_> = windows.into_iter().map(|w| Mutex::new(Some(w))).collect();

            move |block_idx: usize| {
                let window = &windows[block_idx];
                let mut window = window.lock().unwrap().take().unwrap();

                move |idx: &[usize], elem| {
                    window[idx] = elem;
                }
            }
        };

        decoder.decode(writer, &range, &[0, steps, 0]);
        let mut img = image::Rgb32FImage::new(data.dims()[0] as u32, data.dims()[1] as u32);
        for (p, rgb) in img.pixels_mut().zip(data.flatten()) {
            p.0 = *rgb.as_ref();
        }
        let img = image::DynamicImage::ImageRgb32F(img).into_rgb8();
        img.save(res_path.join("img_1_ref_x_0.png")).unwrap();

        for x in 1..steps + 1 {
            let mut next = VolumeBlock::new_zero(&dims).unwrap();
            let reader = |counts: &[usize]| {
                let windows = data.window().divide_into(counts);

                let (windows, _, _) = windows.into_raw_parts();
                let windows: Vec<_> = windows.into_iter().map(|w| Mutex::new(Some(w))).collect();

                move |block_idx: usize| {
                    let window = &windows[block_idx];
                    let window = window.lock().unwrap().take().unwrap();
                    move |idx: &[usize]| window[idx]
                }
            };

            let force_once = ForceOnce;
            let writer = |counts: &[usize]| {
                force_once.consume();

                let windows = next.window_mut().divide_into_mut(counts);

                let (windows, _, _) = windows.into_raw_parts();
                let windows: Vec<_> = windows.into_iter().map(|w| Mutex::new(Some(w))).collect();

                move |block_idx: usize| {
                    let window = &windows[block_idx];
                    let mut window = window.lock().unwrap().take().unwrap();

                    move |idx: &[usize], elem| {
                        window[idx] = elem;
                    }
                }
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
