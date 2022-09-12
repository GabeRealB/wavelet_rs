use num_traits::Num;

use crate::stream::{Deserializable, Serializable};

use super::{Backwards, Forwards, OneWayTransform};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Resample;

impl<N: Num + Lerp + Copy> OneWayTransform<Forwards, N> for Resample {
    type Cfg<'a> = ResampleCfg<'a>;

    #[inline]
    fn apply(
        &self,
        input: crate::volume::VolumeBlock<N>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<N> {
        assert!(input.dims().len() == cfg.to.len());
        if input.dims() == cfg.to {
            return input;
        }

        let mut resampled = crate::volume::VolumeBlock::new(cfg.to).unwrap();
        let input_window = input.window();
        let mut output_window = resampled.window_mut();

        for i in 0..input.dims().len() {
            if input.dims()[i] == cfg.to[i] {
                continue;
            }

            let num_steps = cfg.to[i];
            let step_size = (input.dims()[i] as f32) / (cfg.to[i] as f32);

            let src = input_window.rows(i);
            let dst = output_window.rows_mut(i);

            for (src, mut dst) in src.zip(dst) {
                for j in 0..num_steps {
                    let pos = j as f32 * step_size;
                    let first = pos.floor() as usize;
                    let second = (pos.ceil() as usize).min(src.len() - 1);
                    let t = pos.fract();

                    let first = src[first];
                    let second = src[second];
                    let val = first.lerp(second, t);

                    dst[j] = val;
                }
            }
        }

        resampled
    }
}

impl<N: Num + Lerp + Copy> OneWayTransform<Backwards, N> for Resample {
    type Cfg<'a> = ResampleCfg<'a>;

    #[inline]
    fn apply(
        &self,
        input: crate::volume::VolumeBlock<N>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<N> {
        <Self as OneWayTransform<Forwards, N>>::apply(self, input, cfg)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResampleCfg<'a> {
    to: &'a [usize],
}

impl<'a> ResampleCfg<'a> {
    /// Constructs a new `ResampleCfg`.
    pub fn new(to: &'a [usize]) -> Self {
        Self { to }
    }
}

impl<'a> From<&'a ResampleCfgOwned> for ResampleCfg<'a> {
    fn from(x: &'a ResampleCfgOwned) -> Self {
        Self::new(&x.to)
    }
}

impl Serializable for ResampleCfg<'_> {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.to.serialize(stream)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResampleCfgOwned {
    to: Vec<usize>,
}

impl ResampleCfgOwned {
    /// Constructs a new `ResampleCfgOwned`.
    pub fn new(to: Vec<usize>) -> Self {
        Self { to }
    }
}

impl From<ResampleCfg<'_>> for ResampleCfgOwned {
    fn from(x: ResampleCfg<'_>) -> Self {
        Self::new(x.to.into())
    }
}

impl Serializable for ResampleCfgOwned {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.to.serialize(stream)
    }
}

impl Deserializable for ResampleCfgOwned {
    fn deserialize(stream: &mut crate::stream::DeserializeStream<'_>) -> Self {
        let to = Deserializable::deserialize(stream);
        Self { to }
    }
}

pub trait Lerp<Other = Self> {
    type Output;

    fn lerp(self, other: Other, t: f32) -> Self;
}

impl Lerp for f32 {
    type Output = f32;

    #[inline]
    fn lerp(self, other: Self, t: f32) -> Self {
        debug_assert!((0.0..=1.0).contains(&t));
        ((1.0 - t) * self) + (t * other)
    }
}

impl Lerp for f64 {
    type Output = f64;

    #[inline]
    fn lerp(self, other: Self, t: f32) -> Self {
        debug_assert!((0.0..=1.0).contains(&t));
        let t = t as f64;
        ((1.0 - t) * self) + (t * other)
    }
}
