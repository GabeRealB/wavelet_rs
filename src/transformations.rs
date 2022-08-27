//! Wavelet based transformations.

use std::sync::atomic::AtomicUsize;

use crate::{
    volume::{VolumeBlock, VolumeWindow, VolumeWindowMut},
    wavelet::Wavelet,
};

pub trait Transformation {
    fn forwards(&self, wavelet: &impl Wavelet, input: VolumeBlock, steps: &[u32]) -> VolumeBlock;

    fn backwards(&self, wavelet: &impl Wavelet, input: VolumeBlock, steps: &[u32]) -> VolumeBlock;

    fn forwards_step(
        &self,
        dim: usize,
        wavelet: &impl Wavelet,
        input: &VolumeWindow<'_>,
        low: &mut VolumeWindowMut<'_>,
        high: &mut VolumeWindowMut<'_>,
    );

    fn backwards_step(
        &self,
        dim: usize,
        wavelet: &impl Wavelet,
        output: &mut VolumeWindowMut<'_>,
        low: &VolumeWindow<'_>,
        high: &VolumeWindow<'_>,
    );
}

pub struct WaveletTransform;

impl WaveletTransform {}

pub struct WaveletPacketTransform;

impl WaveletPacketTransform {
    fn forwards_impl(
        &self,
        wavelet: &impl Wavelet,
        input: VolumeWindowMut<'_>,
        output: VolumeWindowMut<'_>,
        ops: &[ForwardsOperation],
        threads: &AtomicUsize,
        max_threads: usize,
    ) {
        if ops.is_empty() {
            return;
        }

        let dim = ops[0].dim;
        let (mut low, mut high) = output.split_into(dim);
        self.forwards_step(dim, wavelet, &input.window(), &mut low, &mut high);

        let (mut input_low, mut input_high) = input.split_into(dim);
        low.copy_to(&mut input_low);
        high.copy_to(&mut input_high);

        if threads.load(std::sync::atomic::Ordering::Relaxed) < max_threads
            && threads.fetch_add(1, std::sync::atomic::Ordering::AcqRel) < max_threads
        {
            std::thread::scope(|scope| {
                let t = scope.spawn(move || {
                    self.forwards_impl(wavelet, input_low, low, &ops[1..], threads, max_threads);
                });

                self.forwards_impl(wavelet, input_high, high, &ops[1..], threads, max_threads);
                t.join().unwrap();
            });
            threads.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        } else {
            self.forwards_impl(wavelet, input_low, low, &ops[1..], threads, max_threads);
            self.forwards_impl(wavelet, input_high, high, &ops[1..], threads, max_threads);
        }
    }

    fn backwards_impl(
        &self,
        wavelet: &impl Wavelet,
        mut input: VolumeWindowMut<'_>,
        mut output: VolumeWindowMut<'_>,
        ops: &[BackwardsOperation],
        threads: &AtomicUsize,
        max_threads: usize,
    ) {
        if ops.is_empty() {
            return;
        }

        #[allow(irrefutable_let_patterns)]
        if let BackwardsOperation::Backwards { dim } = ops[0] {
            let (low, high) = input.split_mut(dim);
            let (output_low, output_high) = output.split_mut(dim);

            if threads.load(std::sync::atomic::Ordering::Relaxed) < max_threads
                && threads.fetch_add(1, std::sync::atomic::Ordering::AcqRel) < max_threads
            {
                std::thread::scope(|scope| {
                    let t = scope.spawn(move || {
                        self.backwards_impl(
                            wavelet,
                            low,
                            output_low,
                            &ops[1..],
                            threads,
                            max_threads,
                        );
                    });

                    self.backwards_impl(
                        wavelet,
                        high,
                        output_high,
                        &ops[1..],
                        threads,
                        max_threads,
                    );
                    t.join().unwrap();
                });
                threads.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            } else {
                self.backwards_impl(wavelet, low, output_low, &ops[1..], threads, max_threads);
                self.backwards_impl(wavelet, high, output_high, &ops[1..], threads, max_threads);
            }

            let (mut low, mut high) = input.split_into(dim);
            self.backwards_step(dim, wavelet, &mut output, &low.window(), &high.window());

            let (output_low, output_high) = output.split_into(dim);
            output_low.copy_to(&mut low);
            output_high.copy_to(&mut high);
        }
    }
}

impl Transformation for WaveletPacketTransform {
    fn forwards(
        &self,
        wavelet: &impl Wavelet,
        mut input: VolumeBlock,
        steps: &[u32],
    ) -> VolumeBlock {
        let dims = input.dims();

        assert!(dims.len() == steps.len());
        for (dim, step) in dims.iter().zip(steps.iter()) {
            let required_size = 2usize.pow(*step);
            assert!(required_size <= *dim);
        }

        if dims.iter().cloned().product::<usize>() == 0 {
            panic!("invalid number of steps");
        }

        if dims.iter().cloned().product::<usize>() == 1 {
            return input;
        }

        let ops = ForwardsOperation::new(steps);
        let mut output = VolumeBlock::new(dims).unwrap();
        let input_window = input.window_mut();
        let output_window = output.window_mut();

        let max_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let threads = AtomicUsize::new(1);

        self.forwards_impl(
            wavelet,
            input_window,
            output_window,
            &ops,
            &threads,
            max_threads,
        );

        output
    }

    fn backwards(
        &self,
        wavelet: &impl Wavelet,
        mut input: VolumeBlock,
        steps: &[u32],
    ) -> VolumeBlock {
        let dims = input.dims();

        assert!(dims.len() == steps.len());
        for (dim, step) in dims.iter().zip(steps.iter()) {
            let required_size = 2usize.pow(*step);
            assert!(required_size <= *dim);
        }

        if dims.iter().cloned().product::<usize>() == 0 {
            panic!("invalid number of steps");
        }

        if dims.iter().cloned().product::<usize>() == 1 {
            return input;
        }

        let ops = BackwardsOperation::new(steps);
        let mut output = VolumeBlock::new(dims).unwrap();
        let input_window = input.window_mut();
        let output_window = output.window_mut();

        let max_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let threads = AtomicUsize::new(1);

        self.backwards_impl(
            wavelet,
            input_window,
            output_window,
            &ops,
            &threads,
            max_threads,
        );

        output
    }

    fn forwards_step(
        &self,
        dim: usize,
        wavelet: &impl Wavelet,
        input: &VolumeWindow<'_>,
        low: &mut VolumeWindowMut<'_>,
        high: &mut VolumeWindowMut<'_>,
    ) {
        let input_rows = input.rows(dim);
        let low_rows = low.rows_mut(dim);
        let high_rows = high.rows_mut(dim);

        for ((input, mut low), mut high) in input_rows.zip(low_rows).zip(high_rows) {
            wavelet.forwards(&input, &mut low, &mut high)
        }
    }

    fn backwards_step(
        &self,
        dim: usize,
        wavelet: &impl Wavelet,
        output: &mut VolumeWindowMut<'_>,
        low: &VolumeWindow<'_>,
        high: &VolumeWindow<'_>,
    ) {
        let output_rows = output.rows_mut(dim);
        let low_rows = low.rows(dim);
        let high_rows = high.rows(dim);

        for ((mut output, low), high) in output_rows.zip(low_rows).zip(high_rows) {
            wavelet.backwards(&low, &high, &mut output)
        }
    }
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
}

#[cfg(test)]
mod tests {
    use crate::{
        volume::VolumeBlock,
        wavelet::{HaarAverageWavelet, HaarWavelet},
    };

    use super::{Transformation, WaveletPacketTransform};

    const TRANSFORM_ERROR: f32 = 0.001;

    #[test]
    fn haar() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dims = [8];
        let block = VolumeBlock::new_with_data(&dims, data).unwrap();
        let block_clone = block.clone();
        println!("Block {:?}", block);

        let transform = WaveletPacketTransform;
        let wavelet = HaarWavelet;

        let transformed = transform.forwards(&wavelet, block, &[3]);
        println!("Transformed {:?}", transformed);

        let backwards = transform.backwards(&wavelet, transformed, &[3]);
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

        let transform = WaveletPacketTransform;
        let wavelet = HaarAverageWavelet;

        let transformed = transform.forwards(&wavelet, block, &[1, 1, 1]);
        println!("Transformed {:?}", transformed);

        let backwards = transform.backwards(&wavelet, transformed, &[1, 1, 1]);
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
        let transform = WaveletPacketTransform;
        let wavelet = HaarAverageWavelet;

        let transformed = transform.forwards(&wavelet, block, &steps);
        let backwards = transform.backwards(&wavelet, transformed, &steps);
        assert!(block_clone.is_equal(&backwards, TRANSFORM_ERROR));
    }
}
