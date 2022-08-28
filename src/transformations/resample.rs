use num_traits::Num;

use super::Transformation;

pub struct Resample<const N: usize> {
    orig: [usize; N],
    resampled: [usize; N],
}

impl<const N: usize> Resample<N> {
    /// Constructs a new `Resample`.
    #[inline]
    pub fn new(from: [usize; N], to: [usize; N]) -> Self {
        Self {
            orig: from,
            resampled: to,
        }
    }

    /// Constructs a new identity `Resample`.
    #[inline]
    pub fn identity(samples: [usize; N]) -> Self {
        Self {
            orig: samples,
            resampled: samples,
        }
    }
}

impl<T, const N: usize> Transformation<T> for Resample<N>
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

        for i in 0..N {
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
        Resample::new(self.resampled, self.orig).forwards(input)
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
