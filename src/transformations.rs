//! Wavelet based transformations.
use crate::volume::VolumeBlock;
use num_traits::Num;

mod basic;
mod resample;
mod wavelet_packet_transform;
mod wavelet_transform;

pub use basic::{Chain, Identity, Reverse};
pub use resample::{Lerp, Resample};
pub use wavelet_packet_transform::WaveletPacketTransform;
pub use wavelet_transform::WaveletTransform;

pub trait Transformation<T: Num + Copy> {
    fn forwards(&self, input: VolumeBlock<T>) -> VolumeBlock<T>;

    fn backwards(&self, input: VolumeBlock<T>) -> VolumeBlock<T>;
}

#[derive(Debug, Clone, Copy)]
struct ForwardsOperation {
    dim: usize,
}

impl ForwardsOperation {
    fn new(steps: &[u32]) -> Vec<Self> {
        let mut ops = Vec::new();
        let mut step = vec![0; steps.len()];

        let mut stop = false;
        while !stop {
            stop = true;

            for (i, (step, max)) in step.iter_mut().zip(steps).enumerate() {
                if *step < *max {
                    *step += 1;
                    stop = false;
                    ops.push(Self { dim: i });
                }
            }
        }

        ops
    }
}

#[derive(Debug, Clone, Copy)]
enum BackwardsOperation {
    Backwards { dim: usize },
}

impl BackwardsOperation {
    fn new(steps: &[u32]) -> Vec<Self> {
        let mut ops = Vec::new();
        let mut step = vec![0; steps.len()];

        let mut stop = false;
        while !stop {
            stop = true;

            for (i, (step, max)) in step.iter_mut().zip(steps).enumerate() {
                if *step < *max {
                    *step += 1;
                    stop = false;
                    ops.push(BackwardsOperation::Backwards { dim: i });
                }
            }
        }

        ops
    }

    fn dim(&self) -> usize {
        match self {
            BackwardsOperation::Backwards { dim } => *dim,
        }
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
        transformations::WaveletTransform,
        vector::Vector,
        volume::VolumeBlock,
        wavelet::{HaarAverageWavelet, HaarWavelet},
    };

    use super::{resample::Resample, Chain, Transformation, WaveletPacketTransform};

    const TRANSFORM_ERROR: f32 = 0.001;

    #[test]
    fn haar() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dims = [8];
        let block = VolumeBlock::new_with_data(&dims, data).unwrap();
        let block_clone = block.clone();
        println!("Block {:?}", block);

        let transform = WaveletTransform::new(HaarWavelet, [3]);

        let transformed = transform.forwards(block);
        println!("Transformed {:?}", transformed);

        let backwards = transform.backwards(transformed);
        println!("Original {:?}", backwards);
        assert!(block_clone.is_equal(&backwards, TRANSFORM_ERROR));
    }

    #[test]
    fn haar_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dims = [2, 2, 2];
        let block = VolumeBlock::new_with_data(&dims, data).unwrap();
        let block_clone = block.clone();
        println!("Block {:?}", block);

        let transform = WaveletTransform::new(HaarAverageWavelet, [1, 1, 1]);

        let transformed = transform.forwards(block);
        println!("Transformed {:?}", transformed);

        let backwards = transform.backwards(transformed);
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
        let transform = WaveletTransform::new(HaarAverageWavelet, steps);

        let transformed = transform.forwards(block);
        let backwards = transform.backwards(transformed);
        assert!(block_clone.is_equal(&backwards, TRANSFORM_ERROR));
    }

    #[test]
    fn image_haar() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img.jpg");
        let img_forwards_path = res_path.join("img_forwards_haar.png");
        let img_backwards_path = res_path.join("img_backwards_haar.png");

        let transform = WaveletTransform::new(HaarWavelet, [2, 2]);
        build_img(
            false,
            transform,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    #[test]
    fn image_haar_average() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img.jpg");
        let img_forwards_path = res_path.join("img_forwards_haar_average.png");
        let img_backwards_path = res_path.join("img_backwards_haar_average.png");

        let transform = WaveletTransform::new(HaarAverageWavelet, [2, 2]);
        build_img(
            false,
            transform,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    #[test]
    fn image_2_haar() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img2.jpg");
        let img_forwards_path = res_path.join("img2_forwards_haar.png");
        let img_backwards_path = res_path.join("img2_backwards_haar.png");

        let transform = WaveletTransform::new(HaarWavelet, [2, 2]);
        build_img(
            true,
            transform,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    #[test]
    fn image_2_haar_average() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img2.jpg");
        let img_forwards_path = res_path.join("img2_forwards_haar_average.png");
        let img_backwards_path = res_path.join("img2_backwards_haar_average.png");

        let transform = WaveletTransform::new(HaarAverageWavelet, [2, 2]);
        build_img(
            true,
            transform,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    #[test]
    fn haar_packet() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dims = [8];
        let block = VolumeBlock::new_with_data(&dims, data).unwrap();
        let block_clone = block.clone();
        println!("Block {:?}", block);

        let transform = WaveletPacketTransform::new(HaarWavelet, [3]);

        let transformed = transform.forwards(block);
        println!("Transformed {:?}", transformed);

        let backwards = transform.backwards(transformed);
        println!("Original {:?}", backwards);
        assert!(block_clone.is_equal(&backwards, TRANSFORM_ERROR));
    }

    #[test]
    fn haar_average_packet() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dims = [2, 2, 2];
        let block = VolumeBlock::new_with_data(&dims, data).unwrap();
        let block_clone = block.clone();
        println!("Block {:?}", block);

        let transform = WaveletPacketTransform::new(HaarAverageWavelet, [1, 1, 1]);

        let transformed = transform.forwards(block);
        println!("Transformed {:?}", transformed);

        let backwards = transform.backwards(transformed);
        println!("Original {:?}", backwards);
        assert!(block_clone.is_equal(&backwards, TRANSFORM_ERROR));
    }

    #[test]
    fn big_block_packet() {
        let dims = [128, 128, 128, 8, 2];
        let elements = dims.iter().product();
        let mut data = Vec::with_capacity(elements);
        for i in 0..elements {
            data.push((i % 100) as f32);
        }

        let block = VolumeBlock::new_with_data(&dims, data).unwrap();
        let block_clone = block.clone();

        let steps = dims.map(|d| d.ilog2());
        let transform = WaveletPacketTransform::new(HaarAverageWavelet, steps);

        let transformed = transform.forwards(block);
        let backwards = transform.backwards(transformed);
        assert!(block_clone.is_equal(&backwards, TRANSFORM_ERROR));
    }

    #[test]
    fn image_haar_package() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img.jpg");
        let img_forwards_path = res_path.join("img_forwards_haar_package.png");
        let img_backwards_path = res_path.join("img_backwards_haar_package.png");

        let transform = WaveletPacketTransform::new(HaarWavelet, [2, 2]);
        build_img(
            false,
            transform,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    #[test]
    fn image_haar_average_package() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img.jpg");
        let img_forwards_path = res_path.join("img_forwards_haar_average_package.png");
        let img_backwards_path = res_path.join("img_backwards_haar_average_package.png");

        let transform = WaveletPacketTransform::new(HaarAverageWavelet, [2, 2]);
        build_img(
            false,
            transform,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    #[test]
    fn image_2_haar_package() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img2.jpg");
        let img_forwards_path = res_path.join("img2_forwards_haar_package.png");
        let img_backwards_path = res_path.join("img2_backwards_haar_package.png");

        let transform = WaveletPacketTransform::new(HaarWavelet, [2, 2]);
        build_img(
            true,
            transform,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    #[test]
    fn image_2_haar_average_package() {
        let mut res_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        res_path.push("resources/test");

        let img_path = res_path.join("img2.jpg");
        let img_forwards_path = res_path.join("img2_forwards_haar_average_package.png");
        let img_backwards_path = res_path.join("img2_backwards_haar_average_package.png");

        let transform = WaveletPacketTransform::new(HaarAverageWavelet, [2, 2]);
        build_img(
            true,
            transform,
            img_path,
            img_forwards_path,
            img_backwards_path,
        );
    }

    fn build_img(
        resample: bool,
        transform: impl Transformation<Vector<f32, 3>>,
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
        let data: Vec<_> = img.pixels().map(|p| Vector::new(p.0)).collect();

        let resample = if resample {
            Resample::new(
                [width, height],
                [width.next_power_of_two(), height.next_power_of_two()],
            )
        } else {
            Resample::identity([width, height])
        };
        let transform = Chain::from((resample, transform));

        let volume_dims = [width, height];
        let volume = VolumeBlock::new_with_data(&volume_dims, data).unwrap();
        let volume = transform.forwards(volume);

        let mut img = image::Rgb32FImage::new(volume.dims()[0] as u32, volume.dims()[1] as u32);
        for (p, rgb) in img.pixels_mut().zip(volume.flatten()) {
            p.0 = *rgb.as_ref();
        }
        let img = image::DynamicImage::ImageRgb32F(img).into_rgb8();
        img.save(img_forwards_path).unwrap();

        let volume = transform.backwards(volume);

        let mut img = image::Rgb32FImage::new(width as u32, height as u32);
        for (p, rgb) in img.pixels_mut().zip(volume.flatten()) {
            p.0 = *rgb.as_ref();
        }
        let img = image::DynamicImage::ImageRgb32F(img).into_rgb8();
        img.save(img_backwards_path).unwrap();
    }
}
