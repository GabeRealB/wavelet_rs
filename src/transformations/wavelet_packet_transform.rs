use std::{marker::PhantomData, sync::atomic::AtomicUsize};

use num_traits::Num;

use crate::{
    transformations::{BackwardsOperation, ForwardsOperation},
    volume::{VolumeBlock, VolumeWindowMut},
    wavelet::{backwards_window, forwards_window, Wavelet},
};

use super::Transformation;

pub struct WaveletPacketTransform<N: Num + Copy + Send, T: Wavelet<N>, const S: usize>(
    T,
    [u32; S],
    PhantomData<fn() -> N>,
);

impl<N: Num + Copy + Send, T: Wavelet<N>, const S: usize> WaveletPacketTransform<N, T, S> {
    /// Constructs a new `WaveletPacketTransform` with the provided wavelet.
    pub fn new(wavelet: T, steps: [u32; S]) -> Self {
        Self(wavelet, steps, PhantomData)
    }

    fn forw_(
        &self,
        input: VolumeWindowMut<'_, N>,
        output: VolumeWindowMut<'_, N>,
        ops: &[ForwardsOperation],
        threads: &AtomicUsize,
        max_threads: usize,
    ) {
        if ops.is_empty() {
            return;
        }

        let dim = ops[0].dim;
        let (mut low, mut high) = output.split_into(dim);
        forwards_window(dim, &self.0, &input.window(), &mut low, &mut high);

        let (mut input_low, mut input_high) = input.split_into(dim);
        low.copy_to(&mut input_low);
        high.copy_to(&mut input_high);

        if threads.load(std::sync::atomic::Ordering::Relaxed) < max_threads
            && threads.fetch_add(1, std::sync::atomic::Ordering::AcqRel) < max_threads
        {
            std::thread::scope(|scope| {
                let t = scope.spawn(move || {
                    self.forw_(input_low, low, &ops[1..], threads, max_threads);
                });

                self.forw_(input_high, high, &ops[1..], threads, max_threads);
                t.join().unwrap();
            });
            threads.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        } else {
            self.forw_(input_low, low, &ops[1..], threads, max_threads);
            self.forw_(input_high, high, &ops[1..], threads, max_threads);
        }
    }

    fn back_(
        &self,
        mut input: VolumeWindowMut<'_, N>,
        mut output: VolumeWindowMut<'_, N>,
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
                        self.back_(low, output_low, &ops[1..], threads, max_threads);
                    });

                    self.back_(high, output_high, &ops[1..], threads, max_threads);
                    t.join().unwrap();
                });
                threads.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            } else {
                self.back_(low, output_low, &ops[1..], threads, max_threads);
                self.back_(high, output_high, &ops[1..], threads, max_threads);
            }

            let (mut low, mut high) = input.split_into(dim);
            backwards_window(dim, &self.0, &mut output, &low.window(), &high.window());

            let (output_low, output_high) = output.split_into(dim);
            output_low.copy_to(&mut low);
            output_high.copy_to(&mut high);
        }
    }
}

impl<N: Num + Copy + Send, T: Wavelet<N>, const S: usize> Transformation<N>
    for WaveletPacketTransform<N, T, S>
{
    fn forwards(&self, mut input: VolumeBlock<N>) -> VolumeBlock<N> {
        let dims = input.dims();

        assert!(dims.len() == self.1.len());
        for (dim, step) in dims.iter().zip(self.1.iter()) {
            let required_size = 2usize.pow(*step);
            assert!(required_size <= *dim);
        }

        if dims.iter().cloned().product::<usize>() == 0 {
            panic!("invalid number of steps");
        }

        if dims.iter().cloned().product::<usize>() == 1 {
            return input;
        }

        let ops = ForwardsOperation::new(&self.1);
        let mut output = VolumeBlock::new(dims).unwrap();
        let input_window = input.window_mut();
        let output_window = output.window_mut();

        let max_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let threads = AtomicUsize::new(1);

        self.forw_(input_window, output_window, &ops, &threads, max_threads);

        output
    }

    fn backwards(&self, mut input: VolumeBlock<N>) -> VolumeBlock<N> {
        let dims = input.dims();

        assert!(dims.len() == self.1.len());
        for (dim, step) in dims.iter().zip(self.1.iter()) {
            let required_size = 2usize.pow(*step);
            assert!(required_size <= *dim);
        }

        if dims.iter().cloned().product::<usize>() == 0 {
            panic!("invalid number of steps");
        }

        if dims.iter().cloned().product::<usize>() == 1 {
            return input;
        }

        let ops = BackwardsOperation::new(&self.1);
        let mut output = VolumeBlock::new(dims).unwrap();
        let input_window = input.window_mut();
        let output_window = output.window_mut();

        let max_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let threads = AtomicUsize::new(1);

        self.back_(input_window, output_window, &ops, &threads, max_threads);

        output
    }
}
