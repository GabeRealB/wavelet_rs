use std::{marker::PhantomData, sync::atomic::AtomicUsize};

use num_traits::Num;

use crate::{
    filter::{backwards_window, forwards_window, upscale_window, Filter},
    stream::{Deserializable, Named, Serializable},
    volume::{VolumeBlock, VolumeWindowMut},
};

use super::{Backwards, BackwardsOperation, Forwards, ForwardsOperation, OneWayTransform};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaveletTransform<N: Num + Copy + Send, T: Filter<N>> {
    filter: T,
    split_high: bool,
    _phantom: PhantomData<fn() -> N>,
}

impl<N: Num + Copy + Send, T: Filter<N>> WaveletTransform<N, T> {
    /// Constructs a new `WaveletTransform` with the provided filter.
    pub fn new(filter: T, split_high: bool) -> Self {
        Self {
            filter,
            split_high,
            _phantom: PhantomData,
        }
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
        let has_high_pass = ops.get(1).map(|o| o.dim > dim).unwrap_or(false);
        let has_high_pass = has_high_pass && self.split_high;
        let (mut low, mut high) = output.split_into(dim);
        forwards_window(dim, &self.filter, &input.window(), &mut low, &mut high);

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

                if has_high_pass {
                    self.forw_high(input_high, high, &ops[1..], dim, (threads, max_threads));
                }
                t.join().unwrap();
            });
            threads.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        } else {
            self.forw_(input_low, low, &ops[1..], threads, max_threads);

            if has_high_pass {
                self.forw_high(input_high, high, &ops[1..], dim, (threads, max_threads));
            }
        }
    }

    fn forw_high(
        &self,
        input: VolumeWindowMut<'_, N>,
        output: VolumeWindowMut<'_, N>,
        ops: &[ForwardsOperation],
        last_dim: usize,
        threads: (&AtomicUsize, usize),
    ) {
        if ops.is_empty() || ops[0].dim < last_dim {
            return;
        }

        let dim = ops[0].dim;
        let (mut low, mut high) = output.split_into(dim);
        forwards_window(dim, &self.filter, &input.window(), &mut low, &mut high);

        let (mut input_low, mut input_high) = input.split_into(dim);
        low.copy_to(&mut input_low);
        high.copy_to(&mut input_high);

        if threads.0.load(std::sync::atomic::Ordering::Relaxed) < threads.1
            && threads.0.fetch_add(1, std::sync::atomic::Ordering::AcqRel) < threads.1
        {
            std::thread::scope(|scope| {
                let t = scope.spawn(move || {
                    self.forw_high(input_low, low, &ops[1..], dim, threads);
                });

                self.forw_high(input_high, high, &ops[1..], dim, threads);
                t.join().unwrap();
            });
            threads.0.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        } else {
            self.forw_high(input_low, low, &ops[1..], dim, threads);
            self.forw_high(input_high, high, &ops[1..], dim, threads);
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

        match ops[0] {
            BackwardsOperation::Backwards { dim } => {
                let has_high_pass = ops.get(1).map(|o| o.dim() > dim).unwrap_or(false);
                let has_high_pass = has_high_pass && self.split_high;

                let (low, high) = input.split_mut(dim);
                let (output_low, output_high) = output.split_mut(dim);

                if threads.load(std::sync::atomic::Ordering::Relaxed) < max_threads
                    && threads.fetch_add(1, std::sync::atomic::Ordering::AcqRel) < max_threads
                {
                    std::thread::scope(|scope| {
                        let t = scope.spawn(move || {
                            self.back_(low, output_low, &ops[1..], threads, max_threads);
                        });

                        if has_high_pass {
                            self.back_high(
                                high,
                                output_high,
                                &ops[1..],
                                dim,
                                (threads, max_threads),
                            );
                        }
                        t.join().unwrap();
                    });
                    threads.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                } else {
                    self.back_(low, output_low, &ops[1..], threads, max_threads);

                    if has_high_pass {
                        self.back_high(high, output_high, &ops[1..], dim, (threads, max_threads));
                    }
                }

                let (mut low, mut high) = input.split_into(dim);
                backwards_window(
                    dim,
                    &self.filter,
                    &mut output,
                    &low.window(),
                    &high.window(),
                );

                let (output_low, output_high) = output.split_into(dim);
                output_low.copy_to(&mut low);
                output_high.copy_to(&mut high);
            }
            BackwardsOperation::Upscale { dim } => {
                let has_high_pass = ops.get(1).map(|o| o.dim() > dim).unwrap_or(false);
                let has_high_pass = has_high_pass && self.split_high;

                let (low, high) = input.split_mut(dim);
                let (output_low, output_high) = output.split_mut(dim);

                if threads.load(std::sync::atomic::Ordering::Relaxed) < max_threads
                    && threads.fetch_add(1, std::sync::atomic::Ordering::AcqRel) < max_threads
                {
                    std::thread::scope(|scope| {
                        let t = scope.spawn(move || {
                            self.back_(low, output_low, &ops[1..], threads, max_threads);
                        });

                        if has_high_pass {
                            self.back_high(
                                high,
                                output_high,
                                &ops[1..],
                                dim,
                                (threads, max_threads),
                            );
                        }
                        t.join().unwrap();
                    });
                    threads.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                } else {
                    self.back_(low, output_low, &ops[1..], threads, max_threads);

                    if has_high_pass {
                        self.back_high(high, output_high, &ops[1..], dim, (threads, max_threads));
                    }
                }

                let (mut low, mut high) = input.split_into(dim);
                upscale_window(dim, &mut output, &low.window());

                let (output_low, output_high) = output.split_into(dim);
                output_low.copy_to(&mut low);
                output_high.copy_to(&mut high);
            }
        }
    }

    fn back_high(
        &self,
        mut input: VolumeWindowMut<'_, N>,
        mut output: VolumeWindowMut<'_, N>,
        ops: &[BackwardsOperation],
        last_dim: usize,
        threads: (&AtomicUsize, usize),
    ) {
        if ops.is_empty() {
            return;
        }

        match ops[0] {
            BackwardsOperation::Backwards { dim } => {
                if dim < last_dim {
                    return;
                }

                let (low, high) = input.split_mut(dim);
                let (output_low, output_high) = output.split_mut(dim);

                if threads.0.load(std::sync::atomic::Ordering::Relaxed) < threads.1
                    && threads.0.fetch_add(1, std::sync::atomic::Ordering::AcqRel) < threads.1
                {
                    std::thread::scope(|scope| {
                        let t = scope.spawn(move || {
                            self.back_high(low, output_low, &ops[1..], dim, threads);
                        });

                        self.back_high(high, output_high, &ops[1..], dim, threads);
                        t.join().unwrap();
                    });
                    threads.0.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                } else {
                    self.back_high(low, output_low, &ops[1..], dim, threads);
                    self.back_high(high, output_high, &ops[1..], dim, threads);
                }

                let (mut low, mut high) = input.split_into(dim);
                backwards_window(
                    dim,
                    &self.filter,
                    &mut output,
                    &low.window(),
                    &high.window(),
                );

                let (output_low, output_high) = output.split_into(dim);
                output_low.copy_to(&mut low);
                output_high.copy_to(&mut high);
            }
            BackwardsOperation::Upscale { dim } => {
                if dim < last_dim {
                    return;
                }

                let (low, high) = input.split_mut(dim);
                let (output_low, output_high) = output.split_mut(dim);

                if threads.0.load(std::sync::atomic::Ordering::Relaxed) < threads.1
                    && threads.0.fetch_add(1, std::sync::atomic::Ordering::AcqRel) < threads.1
                {
                    std::thread::scope(|scope| {
                        let t = scope.spawn(move || {
                            self.back_high(low, output_low, &ops[1..], dim, threads);
                        });

                        self.back_high(high, output_high, &ops[1..], dim, threads);
                        t.join().unwrap();
                    });
                    threads.0.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                } else {
                    self.back_high(low, output_low, &ops[1..], dim, threads);
                    self.back_high(high, output_high, &ops[1..], dim, threads);
                }

                let (mut low, mut high) = input.split_into(dim);
                upscale_window(dim, &mut output, &low.window());

                let (output_low, output_high) = output.split_into(dim);
                output_low.copy_to(&mut low);
                output_high.copy_to(&mut high);
            }
        }
    }
}

impl<N: Num + Copy + Send, T: Filter<N>> OneWayTransform<Forwards, N> for WaveletTransform<N, T> {
    type Cfg<'a> = WaveletDecompCfg<'a>;

    fn apply(&self, mut input: VolumeBlock<N>, cfg: Self::Cfg<'_>) -> VolumeBlock<N> {
        let dims = input.dims();

        assert!(dims.len() == cfg.steps.len());
        for (dim, step) in dims.iter().zip(cfg.steps.iter()) {
            let required_size = 2usize.pow(*step);
            assert!(required_size <= *dim);
        }

        if dims.iter().cloned().product::<usize>() == 0 {
            panic!("invalid number of steps");
        }

        if dims.iter().cloned().product::<usize>() == 1 {
            return input;
        }

        let ops = ForwardsOperation::new(cfg.steps);
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
}

impl<N: Num + Copy + Send, T: Filter<N>> OneWayTransform<Backwards, N> for WaveletTransform<N, T> {
    type Cfg<'a> = WaveletRecompCfg<'a>;

    fn apply(&self, mut input: VolumeBlock<N>, cfg: Self::Cfg<'_>) -> VolumeBlock<N> {
        let dims = input.dims();

        assert!(dims.len() == cfg.backwards.len());
        for (dim, step) in dims.iter().zip(cfg.backwards.iter()) {
            let required_size = 2usize.pow(*step);
            assert!(required_size <= *dim);
        }

        if dims.iter().cloned().product::<usize>() == 0 {
            panic!("invalid number of steps");
        }

        if dims.iter().cloned().product::<usize>() == 1 {
            return input;
        }

        let ops = BackwardsOperation::new(cfg.forwards, cfg.backwards);
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

impl<N: Num + Copy + Send, T: Filter<N> + Serializable> Serializable for WaveletTransform<N, T> {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        N::name().serialize(stream);
        T::name().serialize(stream);
        self.filter.serialize(stream);
        self.split_high.serialize(stream);
    }
}

impl<N: Num + Copy + Send, T: Filter<N> + Deserializable> Deserializable
    for WaveletTransform<N, T>
{
    fn deserialize(stream: &mut crate::stream::DeserializeStream<'_>) -> Self {
        let n_type: String = Deserializable::deserialize(stream);
        let t_type: String = Deserializable::deserialize(stream);
        assert_eq!(n_type, N::name());
        assert_eq!(t_type, T::name());

        let filter = Deserializable::deserialize(stream);
        let split_high = Deserializable::deserialize(stream);
        Self {
            filter,
            split_high,
            _phantom: PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaveletDecompCfg<'a> {
    steps: &'a [u32],
}

impl<'a> WaveletDecompCfg<'a> {
    /// Constructs a new `WaveletDecompCfg`.
    pub fn new(steps: &'a [u32]) -> Self {
        Self { steps }
    }
}

impl<'a> From<&'a WaveletDecompCfgOwned> for WaveletDecompCfg<'a> {
    fn from(x: &'a WaveletDecompCfgOwned) -> Self {
        Self { steps: &x.steps }
    }
}

impl Serializable for WaveletDecompCfg<'_> {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.steps.serialize(stream)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaveletRecompCfg<'a> {
    forwards: &'a [u32],
    backwards: &'a [u32],
}

impl<'a> WaveletRecompCfg<'a> {
    /// Constructs a new `WaveletRecompCfg`.
    pub fn new(forwards: &'a [u32], backwards: &'a [u32]) -> Self {
        Self {
            forwards,
            backwards,
        }
    }
}

impl<'a> From<WaveletDecompCfg<'a>> for WaveletRecompCfg<'a> {
    fn from(x: WaveletDecompCfg<'a>) -> Self {
        Self::new(x.steps, x.steps)
    }
}

impl<'a> From<&'a WaveletRecompCfgOwned> for WaveletRecompCfg<'a> {
    fn from(x: &'a WaveletRecompCfgOwned) -> Self {
        Self {
            forwards: &x.forwards,
            backwards: &x.backwards,
        }
    }
}

impl Serializable for WaveletRecompCfg<'_> {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.forwards.serialize(stream);
        self.backwards.serialize(stream)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaveletDecompCfgOwned {
    steps: Vec<u32>,
}

impl WaveletDecompCfgOwned {
    /// Constructs a new `WaveletDecompCfgOwned`.
    pub fn new(steps: Vec<u32>) -> Self {
        Self { steps }
    }
}

impl From<WaveletDecompCfg<'_>> for WaveletDecompCfgOwned {
    fn from(x: WaveletDecompCfg<'_>) -> Self {
        Self::new(x.steps.into())
    }
}

impl Serializable for WaveletDecompCfgOwned {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.steps.serialize(stream)
    }
}

impl Deserializable for WaveletDecompCfgOwned {
    fn deserialize(stream: &mut crate::stream::DeserializeStream<'_>) -> Self {
        let steps = Deserializable::deserialize(stream);
        Self { steps }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaveletRecompCfgOwned {
    forwards: Vec<u32>,
    backwards: Vec<u32>,
}

impl WaveletRecompCfgOwned {
    /// Constructs a new `WaveletRecompCfgOwned`.
    pub fn new(forwards: Vec<u32>, backwards: Vec<u32>) -> Self {
        Self {
            forwards,
            backwards,
        }
    }
}

impl From<WaveletRecompCfg<'_>> for WaveletRecompCfgOwned {
    fn from(x: WaveletRecompCfg<'_>) -> Self {
        Self::new(x.forwards.into(), x.backwards.into())
    }
}

impl Serializable for WaveletRecompCfgOwned {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.forwards.serialize(stream);
        self.backwards.serialize(stream)
    }
}

impl Deserializable for WaveletRecompCfgOwned {
    fn deserialize(stream: &mut crate::stream::DeserializeStream<'_>) -> Self {
        let forwards = Deserializable::deserialize(stream);
        let backwards = Deserializable::deserialize(stream);
        Self {
            forwards,
            backwards,
        }
    }
}
