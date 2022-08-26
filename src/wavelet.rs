//! Definition of common wavelets.

use crate::volume::{Row, RowMut};

pub trait Wavelet: Sync {
    fn forwards(&self, input: &Row<'_>, low: &mut RowMut<'_>, high: &mut RowMut<'_>);
    fn backwards(&self, low: &Row<'_>, high: &Row<'_>, output: &mut RowMut<'_>);
}

pub struct HaarWavelet;

impl HaarWavelet {
    const VALUE: f32 = std::f32::consts::FRAC_1_SQRT_2;
}

impl Wavelet for HaarWavelet {
    fn forwards(&self, input: &Row<'_>, low: &mut RowMut<'_>, high: &mut RowMut<'_>) {
        for (i, (low, high)) in low.iter().zip(high.iter()).enumerate() {
            let idx_left = (2 * i) % input.len();
            let idx_right = ((2 * i) + 1) % input.len();

            let sum = Self::VALUE * (input[idx_right] + input[idx_left]);
            let diff = Self::VALUE * (input[idx_right] - input[idx_left]);

            *low = sum;
            *high = diff;
        }
    }

    fn backwards(&self, low: &Row<'_>, high: &Row<'_>, output: &mut RowMut<'_>) {
        for (i, (low, high)) in low.iter().zip(high.iter()).enumerate() {
            let idx_left = (2 * i) % output.len();
            let idx_right = ((2 * i) + 1) % output.len();

            let left = Self::VALUE * (low - high);
            let right = Self::VALUE * (low + high);

            output[idx_left] = left;
            output[idx_right] = right;
        }
    }
}

pub struct HaarAverageWavelet;

impl Wavelet for HaarAverageWavelet {
    fn forwards(&self, input: &Row<'_>, low: &mut RowMut<'_>, high: &mut RowMut<'_>) {
        for (i, (low, high)) in low.iter().zip(high.iter()).enumerate() {
            let idx_left = (2 * i) % input.len();
            let idx_right = ((2 * i) + 1) % input.len();

            let average = (input[idx_right] + input[idx_left]) / 2.0;
            let diff = input[idx_left] - average;

            *low = average;
            *high = diff;
        }
    }

    fn backwards(&self, low: &Row<'_>, high: &Row<'_>, output: &mut RowMut<'_>) {
        for (i, (low, high)) in low.iter().zip(high.iter()).enumerate() {
            let idx_left = (2 * i) % output.len();
            let idx_right = ((2 * i) + 1) % output.len();

            let left = high + low;
            let right = (2.0 * low) - left;

            output[idx_left] = left;
            output[idx_right] = right;
        }
    }
}
