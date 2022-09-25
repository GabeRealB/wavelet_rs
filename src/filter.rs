//! Definition of common wavelets.

use num_traits::{Float, FloatConst, Num};

use crate::{
    stream::{Deserializable, Serializable},
    volume::{Row, RowMut, VolumeWindowMut},
};

pub trait Filter<T: Num + Copy>: Sync {
    fn forwards(&self, input: &Row<'_, T>, low: &mut [T], high: &mut [T]);
    fn backwards(&self, output: &mut RowMut<'_, T>, low: &[T], high: &[T]);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HaarWavelet;

impl<T: Float + FloatConst> Filter<T> for HaarWavelet {
    fn forwards(&self, input: &Row<'_, T>, low: &mut [T], high: &mut [T]) {
        let value = T::FRAC_1_SQRT_2();

        for (i, (low, high)) in low.iter_mut().zip(high).enumerate() {
            let idx_left = (2 * i).min(input.len() - 1);
            let idx_right = ((2 * i) + 1).min(input.len() - 1);

            let sum = value * (input[idx_right] + input[idx_left]);
            let diff = value * (input[idx_right] - input[idx_left]);

            *low = sum;
            *high = diff;
        }
    }

    fn backwards(&self, output: &mut RowMut<'_, T>, low: &[T], high: &[T]) {
        let value = T::FRAC_1_SQRT_2();

        for (i, (low, high)) in low.iter().zip(high).enumerate() {
            let idx_left = (2 * i).min(output.len() - 1);
            let idx_right = ((2 * i) + 1).min(output.len() - 1);

            let left = value * (*low - *high);
            let right = value * (*low + *high);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AverageFilter;

impl<T: Float + FloatConst> Filter<T> for AverageFilter {
    fn forwards(&self, input: &Row<'_, T>, low: &mut [T], high: &mut [T]) {
        let two = T::from(2.0).unwrap();

        for (i, (low, high)) in low.iter_mut().zip(high).enumerate() {
            let idx_left = (2 * i).min(input.len() - 1);
            let idx_right = ((2 * i) + 1).min(input.len() - 1);

            let average = (input[idx_right] + input[idx_left]) / two;
            let diff = input[idx_left] - average;

            *low = average;
            *high = diff;
        }
    }

    fn backwards(&self, output: &mut RowMut<'_, T>, low: &[T], high: &[T]) {
        let two = T::from(2.0).unwrap();

        for (i, (low, high)) in low.iter().zip(high).enumerate() {
            let idx_left = (2 * i).min(output.len() - 1);
            let idx_right = ((2 * i) + 1).min(output.len() - 1);

            let left = *high + *low;
            let right = (two * *low) - left;

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

/// Applies the forward procedure on each row of a [`VolumeWindow`]
/// across the dimension `dim`.
pub fn forwards_window<T: Num + Copy>(
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
            *dst = *src;
        }
    }
}

/// Applies the backwards procedure on each row of the low and high pass [`VolumeWindow`]s
/// across the dimension `dim`.
pub fn backwards_window<T: Num + Copy>(
    dim: usize,
    wavelet: &impl Filter<T>,
    output: &mut VolumeWindowMut<'_, T>,
    scratch: &mut [T],
) {
    for mut output in output.rows_mut(dim) {
        let scratch = &mut scratch[..output.len()];
        for (src, dst) in output.iter().zip(scratch.iter_mut()) {
            *dst = *src;
        }

        let (low, high) = scratch.split_at_mut(scratch.len() / 2);
        wavelet.backwards(&mut output, low, high);
    }
}

/// Scales up the data of the low pass by duplicating each element.
pub fn upscale_window<T: Num + Copy>(dim: usize, output: &mut VolumeWindowMut<'_, T>) {
    for mut output in output.rows_mut(dim) {
        let len = output.len() / 2;

        for i in (0..len).rev() {
            let elem = output[i];
            output[2 * i] = elem;
            output[(2 * i) + 1] = elem;
        }
    }
}
