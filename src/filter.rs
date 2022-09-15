//! Definition of common wavelets.

use num_traits::{Float, FloatConst, Num};

use crate::{
    stream::{Deserializable, Serializable},
    volume::{Row, RowMut, VolumeWindow, VolumeWindowMut},
};

pub trait Filter<T: Num + Copy>: Sync {
    fn forwards(&self, input: &Row<'_, T>, low: &mut RowMut<'_, T>, high: &mut RowMut<'_, T>);
    fn backwards(&self, low: &Row<'_, T>, high: &Row<'_, T>, output: &mut RowMut<'_, T>);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HaarWavelet;

impl<T: Float + FloatConst> Filter<T> for HaarWavelet {
    fn forwards(&self, input: &Row<'_, T>, low: &mut RowMut<'_, T>, high: &mut RowMut<'_, T>) {
        let value = T::FRAC_1_SQRT_2();

        for (i, (low, high)) in low.iter().zip(high.iter()).enumerate() {
            let idx_left = (2 * i).min(input.len() - 1);
            let idx_right = ((2 * i) + 1).min(input.len() - 1);

            let sum = value * (input[idx_right] + input[idx_left]);
            let diff = value * (input[idx_right] - input[idx_left]);

            *low = sum;
            *high = diff;
        }
    }

    fn backwards(&self, low: &Row<'_, T>, high: &Row<'_, T>, output: &mut RowMut<'_, T>) {
        let value = T::FRAC_1_SQRT_2();

        for (i, (low, high)) in low.iter().zip(high.iter()).enumerate() {
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
    fn forwards(&self, input: &Row<'_, T>, low: &mut RowMut<'_, T>, high: &mut RowMut<'_, T>) {
        let two = T::from(2.0).unwrap();

        for (i, (low, high)) in low.iter().zip(high.iter()).enumerate() {
            let idx_left = (2 * i).min(input.len() - 1);
            let idx_right = ((2 * i) + 1).min(input.len() - 1);

            let average = (input[idx_right] + input[idx_left]) / two;
            let diff = input[idx_left] - average;

            *low = average;
            *high = diff;
        }
    }

    fn backwards(&self, low: &Row<'_, T>, high: &Row<'_, T>, output: &mut RowMut<'_, T>) {
        let two = T::from(2.0).unwrap();

        for (i, (low, high)) in low.iter().zip(high.iter()).enumerate() {
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
    input: &VolumeWindow<'_, T>,
    low: &mut VolumeWindowMut<'_, T>,
    high: &mut VolumeWindowMut<'_, T>,
) {
    let input_rows = input.rows(dim);
    let low_rows = low.rows_mut(dim);
    let high_rows = high.rows_mut(dim);

    for ((input, mut low), mut high) in input_rows.zip(low_rows).zip(high_rows) {
        wavelet.forwards(&input, &mut low, &mut high)
    }
}

/// Applies the backwards procedure on each row of the low and high pass [`VolumeWindow`]s
/// across the dimension `dim`.
pub fn backwards_window<T: Num + Copy>(
    dim: usize,
    wavelet: &impl Filter<T>,
    output: &mut VolumeWindowMut<'_, T>,
    low: &VolumeWindow<'_, T>,
    high: &VolumeWindow<'_, T>,
) {
    let output_rows = output.rows_mut(dim);
    let low_rows = low.rows(dim);
    let high_rows = high.rows(dim);

    for ((mut output, low), high) in output_rows.zip(low_rows).zip(high_rows) {
        wavelet.backwards(&low, &high, &mut output)
    }
}

/// Scales up the data of the low pass by duplicating each element.
pub fn upscale_window<T: Num + Copy>(
    dim: usize,
    output: &mut VolumeWindowMut<'_, T>,
    low: &VolumeWindow<'_, T>,
) {
    let output_rows = output.rows_mut(dim);
    let low_rows = low.rows(dim);

    for (mut output, low) in output_rows.zip(low_rows) {
        for (i, elem) in low.iter().enumerate() {
            output[2 * i] = *elem;
            output[(2 * i) + 1] = *elem;
        }
    }
}
