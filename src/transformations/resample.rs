use num_traits::Num;

use crate::stream::{Deserializable, Serializable};

use super::Transformation;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Resample {
    orig: Vec<usize>,
    resampled: Vec<usize>,
}

impl Resample {
    /// Constructs a new `Resample`.
    #[inline]
    pub fn new(from: &[usize], to: &[usize]) -> Self {
        assert_eq!(from.len(), to.len());
        Self {
            orig: from.into(),
            resampled: to.into(),
        }
    }

    /// Constructs a new identity `Resample`.
    #[inline]
    pub fn identity(samples: &[usize]) -> Self {
        Self {
            orig: samples.into(),
            resampled: samples.into(),
        }
    }
}

impl<T> Transformation<T> for Resample
where
    T: Num + Lerp + Copy,
{
    #[inline]
    fn forwards(&self, input: crate::volume::VolumeBlock<T>) -> crate::volume::VolumeBlock<T> {
        assert_eq!(input.dims(), self.orig);

        if self.orig == self.resampled {
            return input;
        }

        let mut resampled = crate::volume::VolumeBlock::new(&self.resampled).unwrap();
        let input_window = input.window();
        let mut output_window = resampled.window_mut();

        for i in 0..self.orig.len() {
            let num_steps = self.resampled[i];
            let step_size = (self.orig[i] as f32) / (self.resampled[i] as f32);

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

    #[inline]
    fn backwards(&self, input: crate::volume::VolumeBlock<T>) -> crate::volume::VolumeBlock<T> {
        Resample::new(&self.resampled, &self.orig).forwards(input)
    }
}

impl Serializable for Resample {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.orig.serialize(stream);
        self.resampled.serialize(stream);
    }
}

impl Deserializable for Resample {
    fn deserialize(stream: &mut crate::stream::DeserializeStream<'_>) -> Self {
        let orig = Deserializable::deserialize(stream);
        let resampled = Deserializable::deserialize(stream);

        Self { orig, resampled }
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
