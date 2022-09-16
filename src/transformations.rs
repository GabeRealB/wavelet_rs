//! Wavelet based transformations.
use crate::volume::VolumeBlock;
use num_traits::Num;

mod basic;
mod resample;
pub(crate) mod wavelet_transform;

pub use basic::{Chain, Identity, Reverse};
pub use resample::{Lerp, ResampleCfg, ResampleCfgOwned, ResampleExtend, ResampleLinear};
pub use wavelet_transform::{
    WaveletDecompCfg, WaveletDecompCfgOwned, WaveletRecompCfg, WaveletRecompCfgOwned,
    WaveletTransform,
};

#[derive(Debug)]
pub struct Forwards;

#[derive(Debug)]
pub struct Backwards;

pub trait OneWayTransform<T, N: Num + Copy> {
    type Cfg<'a>;

    fn apply(&self, input: VolumeBlock<N>, cfg: Self::Cfg<'_>) -> VolumeBlock<N>;
}

pub trait ReversibleTransform<N: Num + Copy>:
    OneWayTransform<Forwards, N> + OneWayTransform<Backwards, N>
{
    fn forwards(
        &self,
        input: VolumeBlock<N>,
        cfg: <Self as OneWayTransform<Forwards, N>>::Cfg<'_>,
    ) -> VolumeBlock<N>;

    fn backwards(
        &self,
        input: VolumeBlock<N>,
        cfg: <Self as OneWayTransform<Backwards, N>>::Cfg<'_>,
    ) -> VolumeBlock<N>;
}

impl<T, N> ReversibleTransform<N> for T
where
    N: Num + Copy,
    T: OneWayTransform<Forwards, N> + OneWayTransform<Backwards, N>,
{
    fn forwards(
        &self,
        input: VolumeBlock<N>,
        cfg: <Self as OneWayTransform<Forwards, N>>::Cfg<'_>,
    ) -> VolumeBlock<N> {
        <Self as OneWayTransform<Forwards, N>>::apply(self, input, cfg)
    }

    fn backwards(
        &self,
        input: VolumeBlock<N>,
        cfg: <Self as OneWayTransform<Backwards, N>>::Cfg<'_>,
    ) -> VolumeBlock<N> {
        <Self as OneWayTransform<Backwards, N>>::apply(self, input, cfg)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::BufReader,
        path::{Path, PathBuf},
    };

    use crate::{
        filter::{AverageFilter, Filter, HaarWavelet},
        transformations::{wavelet_transform::WaveletDecompCfg, WaveletTransform},
        vector::Vector,
        volume::VolumeBlock,
    };

    use super::{
        resample::{ResampleCfg, ResampleExtend},
        wavelet_transform::{RefinementInfo, WaveletRecompCfg},
        Chain, ReversibleTransform,
    };

    const TRANSFORM_ERROR: f32 = 0.001;

    #[test]
    fn haar() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dims = [8];
        let block = VolumeBlock::new_with_data(&dims, data).unwrap();
        let block_clone = block.clone();
        println!("Block {:?}", block);

        let f_cfg = WaveletDecompCfg::new(&[3]);
        let b_cfg = f_cfg.into();
        let transform = WaveletTransform::new(HaarWavelet, true);

        let transformed = transform.forwards(block, f_cfg);
        println!("Transformed {:?}", transformed);

        let backwards = transform.backwards(transformed, b_cfg);
        println!("Original {:?}", backwards);
        assert!(block_clone.is_equal(&backwards, TRANSFORM_ERROR));
    }

    #[test]
    fn average_filter() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dims = [2, 2, 2];
        let block = VolumeBlock::new_with_data(&dims, data).unwrap();
        let block_clone = block.clone();
        println!("Block {:?}", block);

        let f_cfg = WaveletDecompCfg::new(&[1, 1, 1]);
        let b_cfg = f_cfg.into();
        let transform = WaveletTransform::new(AverageFilter, true);

        let transformed = transform.forwards(block, f_cfg);
        println!("Transformed {:?}", transformed);

        let backwards = transform.backwards(transformed, b_cfg);
        println!("Original {:?}", backwards);
        assert!(block_clone.is_equal(&backwards, TRANSFORM_ERROR));
    }

    #[test]
    fn big_block() {
        let dims = [128, 128, 128, 8, 2];
        let elements = dims.iter().product();
        let mut data = Vec::with_capacity(elements);
        for i in 0..elements {
            data.push((i % 100) as f32);
        }

        let block = VolumeBlock::new_with_data(&dims, data).unwrap();
        let block_clone = block.clone();

        let steps = dims.map(|d| d.ilog2());
        let f_cfg = WaveletDecompCfg::new(&steps);
        let b_cfg = f_cfg.into();
        let transform = WaveletTransform::new(AverageFilter, true);

        let transformed = transform.forwards(block, f_cfg);
        let backwards = transform.backwards(transformed, b_cfg);
        assert!(block_clone.is_equal(&backwards, TRANSFORM_ERROR));
    }

    #[test]
    fn image_haar() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img_1.jpg");
        let img_forwards_path = res_path.join("img_1_forwards_haar.png");
        let img_backwards_path = res_path.join("img_1_backwards_haar.png");

        let f_cfg = WaveletDecompCfg::new(&[2, 2]);
        let b_cfg = f_cfg.into();
        let transform = WaveletTransform::new(HaarWavelet, true);
        build_img(
            false,
            transform,
            f_cfg,
            b_cfg,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    #[test]
    fn image_average_filter() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img_1.jpg");
        let img_forwards_path = res_path.join("img_1_forwards_average_filter.png");
        let img_backwards_path = res_path.join("img_1_backwards_average_filter.png");

        let f_cfg = WaveletDecompCfg::new(&[2, 2]);
        let b_cfg = f_cfg.into();
        let transform = WaveletTransform::new(AverageFilter, true);
        build_img(
            false,
            transform,
            f_cfg,
            b_cfg,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    #[test]
    fn image_2_haar() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img_2.jpg");
        let img_forwards_path = res_path.join("img_2_forwards_haar.png");
        let img_backwards_path = res_path.join("img_2_backwards_haar.png");

        let f_cfg = WaveletDecompCfg::new(&[2, 2]);
        let b_cfg = f_cfg.into();
        let transform = WaveletTransform::new(HaarWavelet, true);
        build_img(
            true,
            transform,
            f_cfg,
            b_cfg,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    #[test]
    fn image_2_average_filter() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img_2.jpg");
        let img_forwards_path = res_path.join("img_2_forwards_average_filter.png");
        let img_backwards_path = res_path.join("img_2_backwards_average_filter.png");

        let f_cfg = WaveletDecompCfg::new(&[2, 2]);
        let b_cfg = f_cfg.into();
        let transform = WaveletTransform::new(AverageFilter, true);
        build_img(
            true,
            transform,
            f_cfg,
            b_cfg,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    #[test]
    fn image_haar_custom_steps() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img_1.jpg");
        let img_forwards_path = res_path.join("img_1_forwards_haar_custom_steps.png");
        let img_backwards_path = res_path.join("img_1_backwards_haar_custom_steps.png");

        let f_cfg = WaveletDecompCfg::new(&[2, 2]);
        let b_cfg = WaveletRecompCfg::new(&[2, 2], &[1, 2]);
        let transform = WaveletTransform::new(HaarWavelet, false);
        build_img(
            false,
            transform,
            f_cfg,
            b_cfg,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    #[test]
    fn image_average_filter_custom_steps() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img_1.jpg");
        let img_forwards_path = res_path.join("img_1_forwards_average_filter_custom_steps.png");
        let img_backwards_path = res_path.join("img_1_backwards_average_filter_custom_steps.png");

        let f_cfg = WaveletDecompCfg::new(&[2, 2]);
        let b_cfg = WaveletRecompCfg::new(&[2, 2], &[1, 2]);
        let transform = WaveletTransform::new(AverageFilter, false);
        build_img(
            false,
            transform,
            f_cfg,
            b_cfg,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    #[test]
    fn image_2_haar_custom_steps() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img_2.jpg");
        let img_forwards_path = res_path.join("img_2_forwards_haar_custom_steps.png");
        let img_backwards_path = res_path.join("img_2_backwards_haar_custom_steps.png");

        let f_cfg = WaveletDecompCfg::new(&[2, 2]);
        let b_cfg = WaveletRecompCfg::new(&[2, 2], &[1, 2]);
        let transform = WaveletTransform::new(HaarWavelet, false);
        build_img(
            true,
            transform,
            f_cfg,
            b_cfg,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    #[test]
    fn image_2_average_filter_custom_steps() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img_2.jpg");
        let img_forwards_path = res_path.join("img_2_forwards_average_filter_custom_steps.png");
        let img_backwards_path = res_path.join("img_2_backwards_average_filter_custom_steps.png");

        let f_cfg = WaveletDecompCfg::new(&[2, 2]);
        let b_cfg = WaveletRecompCfg::new(&[2, 2], &[1, 2]);
        let transform = WaveletTransform::new(AverageFilter, false);
        build_img(
            true,
            transform,
            f_cfg,
            b_cfg,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    #[test]
    fn image_haar_refinements() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img_1.jpg");
        let f_cfg = WaveletDecompCfg::new(&[2, 2]);
        let transform = WaveletTransform::new(HaarWavelet, false);
        build_img_refinement(
            false,
            transform,
            f_cfg,
            img_path,
            "img_1_decom_haar",
            &[
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [0, 2],
                [1, 1],
                [1, 2],
                [2, 2],
            ],
        );
    }

    #[test]
    fn image_average_filter_refinements() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img_1.jpg");
        let f_cfg = WaveletDecompCfg::new(&[2, 2]);
        let transform = WaveletTransform::new(AverageFilter, false);
        build_img_refinement(
            false,
            transform,
            f_cfg,
            img_path,
            "img_1_decom_average_filter",
            &[
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [0, 2],
                [1, 1],
                [1, 2],
                [2, 2],
            ],
        );
    }

    #[test]
    fn image_2_haar_refinements() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img_2.jpg");
        let f_cfg = WaveletDecompCfg::new(&[2, 2]);
        let transform = WaveletTransform::new(HaarWavelet, false);
        build_img_refinement(
            true,
            transform,
            f_cfg,
            img_path,
            "img_2_decom_haar",
            &[
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [0, 2],
                [1, 1],
                [1, 2],
                [2, 2],
            ],
        );
    }

    #[test]
    fn image_2_average_filter_refinements() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img_2.jpg");
        let f_cfg = WaveletDecompCfg::new(&[2, 2]);
        let transform = WaveletTransform::new(AverageFilter, false);
        build_img_refinement(
            true,
            transform,
            f_cfg,
            img_path,
            "img_2_decom_average_filter",
            &[
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [0, 2],
                [1, 1],
                [1, 2],
                [2, 2],
            ],
        );
    }

    fn build_img<T: Filter<Vector<f32, 3>>>(
        resample: bool,
        transform: WaveletTransform<Vector<f32, 3>, T>,
        f_cfg: WaveletDecompCfg<'_>,
        b_cfg: WaveletRecompCfg<'_>,
        img_path: impl AsRef<Path>,
        img_forwards_path: impl AsRef<Path>,
        img_backwards_path: impl AsRef<Path>,
    ) {
        let f = File::open(img_path).unwrap();
        let reader = BufReader::new(f);
        let img = image::load(reader, image::ImageFormat::Jpeg)
            .unwrap()
            .to_rgb32f();

        let (width, height) = (img.width() as usize, img.height() as usize);
        let (r_width, r_height) = if resample {
            (width.next_power_of_two(), height.next_power_of_two())
        } else {
            (width, height)
        };

        let dims = [width, height];
        let r_dims = [r_width, r_height];
        let data: Vec<_> = img.pixels().map(|p| Vector::new(p.0)).collect();

        let f_cfg = Chain::combine(ResampleCfg::new(&r_dims), f_cfg);
        let b_cfg = Chain::combine(ResampleCfg::new(&dims), b_cfg);
        let transform = Chain::from((ResampleExtend, transform));

        let volume_dims = [width, height];
        let volume = VolumeBlock::new_with_data(&volume_dims, data).unwrap();
        let volume = transform.forwards(volume, f_cfg);

        let mut img = image::Rgb32FImage::new(volume.dims()[0] as u32, volume.dims()[1] as u32);
        for (p, rgb) in img.pixels_mut().zip(volume.flatten()) {
            p.0 = *rgb.as_ref();
        }
        let img = image::DynamicImage::ImageRgb32F(img).into_rgb8();
        img.save(img_forwards_path).unwrap();

        let volume = transform.backwards(volume, b_cfg);

        let mut img = image::Rgb32FImage::new(width as u32, height as u32);
        for (p, rgb) in img.pixels_mut().zip(volume.flatten()) {
            p.0 = *rgb.as_ref();
        }
        let img = image::DynamicImage::ImageRgb32F(img).into_rgb8();
        img.save(img_backwards_path).unwrap();
    }

    fn build_img_refinement<T: Filter<Vector<f32, 3>> + Clone>(
        resample: bool,
        transform: WaveletTransform<Vector<f32, 3>, T>,
        f_cfg: WaveletDecompCfg<'_>,
        img_path: impl AsRef<Path>,
        img_name: &str,
        refinements: &[[u32; 2]],
    ) {
        let f = File::open(&img_path).unwrap();
        let reader = BufReader::new(f);
        let img = image::load(reader, image::ImageFormat::Jpeg)
            .unwrap()
            .to_rgb32f();

        let (width, height) = (img.width() as usize, img.height() as usize);
        let (r_width, r_height) = if resample {
            (width.next_power_of_two(), height.next_power_of_two())
        } else {
            (width, height)
        };

        let dims = [width, height];
        let r_dims = [r_width, r_height];
        let data: Vec<_> = img.pixels().map(|p| Vector::new(p.0)).collect();

        let steps = f_cfg.steps();
        let b_cfg = Chain::combine(ResampleCfg::new(&dims), f_cfg.into());
        let f_cfg = Chain::combine(ResampleCfg::new(&r_dims), f_cfg);
        let f_trans = Chain::from((ResampleExtend, transform.clone()));

        let volume_dims = [width, height];
        let volume = VolumeBlock::new_with_data(&volume_dims, data).unwrap();
        let volume = f_trans.forwards(volume, f_cfg);

        let volume_backwards = f_trans.backwards(volume.clone(), b_cfg);

        for &[x, y] in refinements {
            let info = RefinementInfo::new(&r_dims, steps, &[x, y]);
            let mut block = volume.clone();
            WaveletTransform::<Vector<f32, 3>, T>::adapt_for_refinement(&mut block, &info);

            let p = img_path
                .as_ref()
                .parent()
                .unwrap()
                .join(format!("{img_name}_x_{x}_y_{y}.png"));
            let mut img = image::Rgb32FImage::new(block.dims()[0] as u32, block.dims()[1] as u32);
            for (p, rgb) in img.pixels_mut().zip(block.flatten()) {
                p.0 = *rgb.as_ref();
            }
            let img = image::DynamicImage::ImageRgb32F(img).into_rgb8();
            img.save(p).unwrap();

            let b = [x, y];
            let b_cfg = WaveletRecompCfg::new(steps, &b);
            let vol_b = transform.backwards(volume.clone(), b_cfg);

            let step_x = 2usize.pow(steps[0] - x);
            let step_y = 2usize.pow(steps[1] - y);
            for x in (0..r_width).step_by(step_x) {
                for y in (0..r_height).step_by(step_y) {
                    block[[x / step_x, y / step_y].as_ref()] = vol_b[[x, y].as_ref()];
                }
            }

            let remaining_steps = [steps[0] - x, steps[1] - y];
            let b_cfg = Chain::combine(
                ResampleCfg::new(&dims),
                WaveletRecompCfg::new(&remaining_steps, &remaining_steps),
            );
            let block = f_trans.backwards(block, b_cfg);

            let p = img_path
                .as_ref()
                .parent()
                .unwrap()
                .join(format!("{img_name}_backwards_x_{x}_y_{y}.png"));
            let mut img = image::Rgb32FImage::new(block.dims()[0] as u32, block.dims()[1] as u32);
            for (p, rgb) in img.pixels_mut().zip(block.flatten()) {
                p.0 = *rgb.as_ref();
            }
            let img = image::DynamicImage::ImageRgb32F(img).into_rgb8();
            img.save(p).unwrap();

            let p = img_path
                .as_ref()
                .parent()
                .unwrap()
                .join(format!("{img_name}_backwards_diff_x_{x}_y_{y}.png"));
            let mut img = image::Rgb32FImage::new(block.dims()[0] as u32, block.dims()[1] as u32);
            for ((p, &rgb), &rgb_orig) in img
                .pixels_mut()
                .zip(block.flatten())
                .zip(volume_backwards.flatten())
            {
                p.0 = *(rgb_orig - rgb).as_ref();
            }
            let img = image::DynamicImage::ImageRgb32F(img).into_rgb8();
            img.save(p).unwrap();
        }
    }
}
