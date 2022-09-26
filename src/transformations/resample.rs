use num_traits::Zero;

use super::{Backwards, Forwards, OneWayTransform};
use crate::stream::{Deserializable, Serializable};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResampleIScale;

impl<T: Zero + Clone> OneWayTransform<Forwards, T> for ResampleIScale {
    type Cfg<'a> = ResampleCfg<'a>;

    fn apply(
        &self,
        input: crate::volume::VolumeBlock<T>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<T> {
        assert!(input.dims().len() == cfg.to.len());
        assert!(input
            .dims()
            .iter()
            .zip(cfg.to)
            .all(|(&s, &d)| s <= d && (d % s == 0)));

        if input.dims() == cfg.to {
            return input;
        }

        let mut scaled = crate::volume::VolumeBlock::new_zero(cfg.to).unwrap();
        let mut scaled_window = scaled.window_mut();
        let input_window = input.window();
        input_window.clone_to(&mut scaled_window.custom_range_mut(input_window.dims()));

        for (dim, (&src, &dst)) in input.dims().iter().zip(cfg.to).enumerate() {
            let scale_factor = dst / src;

            for mut row in scaled_window.rows_mut(dim) {
                let len = row.len() / scale_factor;
                for (src, dst) in (0..len).zip((0..row.len()).step_by(scale_factor)).rev() {
                    for i in 0..scale_factor {
                        unsafe {
                            *row.get_unchecked_mut(dst + i) = row.get_unchecked_mut(src).clone()
                        };
                    }
                }
            }
        }

        scaled
    }
}

impl<T: Zero + Clone> OneWayTransform<Backwards, T> for ResampleIScale {
    type Cfg<'a> = ResampleCfg<'a>;

    fn apply(
        &self,
        mut input: crate::volume::VolumeBlock<T>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<T> {
        assert!(input.dims().len() == cfg.to.len());
        assert!(input
            .dims()
            .iter()
            .zip(cfg.to)
            .all(|(&s, &d)| s >= d && (s % d == 0)));

        if input.dims() == cfg.to {
            return input;
        }

        let dims: Vec<_> = input.dims().into();
        let mut window = input.window_mut();

        for (dim, (&src, &dst)) in dims.iter().zip(cfg.to).enumerate() {
            let scale_factor = src / dst;

            for mut row in window.rows_mut(dim) {
                let len = row.len() / scale_factor;
                for (src, dst) in (0..row.len()).step_by(scale_factor).zip(0..len) {
                    unsafe { *row.get_unchecked_mut(dst) = row.get_unchecked_mut(src).clone() };
                }
            }
        }

        let window = window.custom_range(cfg.to);

        let mut output = crate::volume::VolumeBlock::new_zero(cfg.to).unwrap();
        window.clone_to(&mut output.window_mut());

        output
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResampleExtend;

impl<T: Zero + Clone> OneWayTransform<Forwards, T> for ResampleExtend {
    type Cfg<'a> = ResampleCfg<'a>;

    #[inline]
    fn apply(
        &self,
        input: crate::volume::VolumeBlock<T>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<T> {
        assert!(input.dims().len() == cfg.to.len());
        assert!(input.dims().iter().zip(cfg.to).all(|(&s, &d)| s <= d));

        if input.dims() == cfg.to {
            return input;
        }

        let mut resampled = crate::volume::VolumeBlock::new_zero(cfg.to).unwrap();
        let input_window = input.window();

        let mut output_window = resampled.window_mut();
        let mut output_window = output_window.custom_range_mut(input.dims());

        input_window.clone_to(&mut output_window);
        resampled
    }
}

impl<T: Zero + Clone> OneWayTransform<Backwards, T> for ResampleExtend {
    type Cfg<'a> = ResampleCfg<'a>;

    #[inline]
    fn apply(
        &self,
        input: crate::volume::VolumeBlock<T>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<T> {
        assert!(input.dims().len() == cfg.to.len());
        assert!(input.dims().iter().zip(cfg.to).all(|(&s, &d)| s >= d));

        if input.dims() == cfg.to {
            return input;
        }

        let mut resampled = crate::volume::VolumeBlock::new_zero(cfg.to).unwrap();
        let input_window = input.window();
        let input_window = input_window.custom_range(cfg.to);

        let mut output_window = resampled.window_mut();

        input_window.clone_to(&mut output_window);
        resampled
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResampleLinear;

impl<T: Zero + Lerp + Clone> OneWayTransform<Forwards, T> for ResampleLinear {
    type Cfg<'a> = ResampleCfg<'a>;

    #[inline]
    fn apply(
        &self,
        input: crate::volume::VolumeBlock<T>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<T> {
        assert!(input.dims().len() == cfg.to.len());
        if input.dims() == cfg.to {
            return input;
        }

        let mut resampled = crate::volume::VolumeBlock::new_zero(cfg.to).unwrap();
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

                    let first = src[first].clone();
                    let second = src[second].clone();
                    let val = first.lerp(second, t);

                    dst[j] = val;
                }
            }
        }

        resampled
    }
}

impl<T: Zero + Lerp + Clone> OneWayTransform<Backwards, T> for ResampleLinear {
    type Cfg<'a> = ResampleCfg<'a>;

    #[inline]
    fn apply(
        &self,
        input: crate::volume::VolumeBlock<T>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<T> {
        <Self as OneWayTransform<Forwards, T>>::apply(self, input, cfg)
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
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
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
