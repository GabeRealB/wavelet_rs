//! Definition of common wavelets.

use std::ops::{Add, Mul, Sub};

use num_traits::FloatConst;

use crate::{
    stream::{Deserializable, Serializable},
    volume::{Row, RowMut, VolumeWindowMut},
};

/// Trait for implementing filters.
pub trait Filter<T>: Sync {
    /// Splits the input data into a low pass and a high pass.
    fn forwards(&self, input: &Row<'_, T>, low: &mut [T], high: &mut [T]);

    /// Combines the low pass and the high pass into the original data.
    fn backwards(&self, output: &mut RowMut<'_, T>, low: &[T], high: &[T]);
}

/// Filter implementing an Haar wavelet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HaarWavelet;

impl<T> Filter<T> for HaarWavelet
where
    T: PartialOrd + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Clone + FloatConst,
{
    fn forwards(&self, input: &Row<'_, T>, low: &mut [T], high: &mut [T]) {
        let value = T::FRAC_1_SQRT_2();

        for (i, (low, high)) in low.iter_mut().zip(high).enumerate() {
            let idx_left = (2 * i).min(input.len() - 1);
            let idx_right = ((2 * i) + 1).min(input.len() - 1);

            let sum = value.clone() * (input[idx_right].clone() + input[idx_left].clone());
            let diff = value.clone() * (input[idx_right].clone() - input[idx_left].clone());

            *low = sum;
            *high = diff;
        }
    }

    fn backwards(&self, output: &mut RowMut<'_, T>, low: &[T], high: &[T]) {
        let value = T::FRAC_1_SQRT_2();

        for (i, (low, high)) in low.iter().zip(high).enumerate() {
            let idx_left = (2 * i).min(output.len() - 1);
            let idx_right = ((2 * i) + 1).min(output.len() - 1);

            let left = value.clone() * (low.clone() - high.clone());
            let right = value.clone() * (low.clone() + high.clone());

            output[idx_left] = left;
            output[idx_right] = right;
        }
    }
}

impl Serializable for HaarWavelet {
    fn serialize(self, _stream: &mut crate::stream::SerializeStream) {}
}

impl Deserializable for HaarWavelet {
    fn deserialize(_stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        Self
    }
}

/// Variation of the haar wavelet, using the average of pairs as it's low pass filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AverageFilter;

impl<T> Filter<T> for AverageFilter
where
    T: PartialOrd + Add<Output = T> + Sub<Output = T> + Average<Output = T> + Clone + FloatConst,
{
    fn forwards(&self, input: &Row<'_, T>, low: &mut [T], high: &mut [T]) {
        for (i, (low, high)) in low.iter_mut().zip(high).enumerate() {
            let idx_left = (2 * i).min(input.len() - 1);
            let idx_right = ((2 * i) + 1).min(input.len() - 1);

            let average = input[idx_right].clone().avg(input[idx_left].clone());
            let diff = input[idx_left].clone() - average.clone();

            *low = average;
            *high = diff;
        }
    }

    fn backwards(&self, output: &mut RowMut<'_, T>, low: &[T], high: &[T]) {
        for (i, (low, high)) in low.iter().zip(high).enumerate() {
            let idx_left = (2 * i).min(output.len() - 1);
            let idx_right = ((2 * i) + 1).min(output.len() - 1);

            let left = low.clone() + high.clone();
            let right = low.clone() - high.clone();

            output[idx_left] = left;
            output[idx_right] = right;
        }
    }
}

impl Serializable for AverageFilter {
    fn serialize(self, _stream: &mut crate::stream::SerializeStream) {}
}

impl Deserializable for AverageFilter {
    fn deserialize(_stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        Self
    }
}

/// Trait or types implementing an averaging operation.
pub trait Average<Rhs = Self> {
    /// Output type of the operation.
    type Output;

    /// Computes the average of two elements.
    fn avg(self, rhs: Rhs) -> Self::Output;
}

macro_rules! impl_avg_int {
    ($($T:ty),*) => {
        $(
            impl Average for $T {
                type Output = Self;

                #[inline(always)]
                fn avg(self, rhs: Self) -> Self::Output {
                    (self & rhs) + ((self ^ rhs) >> 1)
                }
            }
        )*
    };
}

macro_rules! impl_avg_float {
    ($($T:ty),*) => {
        $(
            impl Average for $T {
                type Output = Self;

                #[inline(always)]
                fn avg(self, rhs: Self) -> Self::Output {
                    (self + rhs) / 2.0
                }
            }
        )*
    };
}

impl_avg_int! {
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
}

impl_avg_float! {
    f32, f64
}

/// Applies the forward procedure on each lane of a [`VolumeWindowMut`]
/// across the dimension `dim` and writes the result back into the lane.
pub fn forwards_window<T: Clone>(
    dim: usize,
    wavelet: &impl Filter<T>,
    input: &mut VolumeWindowMut<'_, T>,
    scratch: &mut [T],
) {
    for mut input in input.rows_mut(dim) {
        let scratch = &mut scratch[..input.len()];
        let (low, high) = scratch.split_at_mut(scratch.len() / 2);
        wavelet.forwards(&input.as_row(), low, high);

        for (src, dst) in scratch.iter_mut().zip(input.into_iter()) {
            *dst = src.clone();
        }
    }
}

/// Applies the backwards procedure on each lane of the low and high
/// pass [`VolumeWindowMut`] across the dimension `dim` and writes
/// the result back into the lane.
pub fn backwards_window<T: Clone>(
    dim: usize,
    wavelet: &impl Filter<T>,
    output: &mut VolumeWindowMut<'_, T>,
    scratch: &mut [T],
) {
    for mut output in output.rows_mut(dim) {
        let scratch = &mut scratch[..output.len()];
        for (src, dst) in output.iter_mut().zip(scratch.iter_mut()) {
            *dst = src.clone();
        }

        let (low, high) = scratch.split_at_mut(scratch.len() / 2);
        wavelet.backwards(&mut output, low, high);
    }
}

/// Scales up the data of the low pass by duplicating each element.
pub fn upscale_window<T: Clone>(dim: usize, output: &mut VolumeWindowMut<'_, T>) {
    for mut output in output.rows_mut(dim) {
        let len = output.len() / 2;

        for i in (0..len).rev() {
            let elem = output[i].clone();
            output[2 * i] = elem.clone();
            output[(2 * i) + 1] = elem;
        }
    }
}
