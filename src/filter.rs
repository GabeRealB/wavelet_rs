//! Definition of common wavelets.

use std::ops::{Add, Mul, Neg, Sub};

use num_traits::{Float, FloatConst, NumCast, Zero};

use crate::{
    stream::{Deserializable, Serializable},
    volume::{Lane, LaneMut, VolumeWindowMut},
};

/// Trait for implementing filters.
pub trait Filter<T>: Sync {
    /// Splits the input data into a low pass and a high pass.
    fn forwards(&self, input: &Lane<'_, T>, low: &mut [T], high: &mut [T]);

    /// Combines the low pass and the high pass into the original data.
    fn backwards(&self, output: &mut LaneMut<'_, T>, low: &[T], high: &[T]);
}

/// Trait to type-erase a filter into a GenericFilter.
pub trait ToGenericFilter<T> {
    /// Type-erases the filter.
    fn to_generic(&self) -> GenericFilter<T>;
}

/// Generic filter.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GenericFilter<T> {
    f_coeff_low: Vec<T>,
    f_coeff_high: Vec<T>,
    b_coeff_low: Vec<T>,
    b_coeff_high: Vec<T>,
}

impl<T> GenericFilter<T> {
    /// Constructs a new generic filter.
    pub fn new(
        f_coeff_low: Vec<T>,
        f_coeff_high: Vec<T>,
        b_coeff_low: Vec<T>,
        b_coeff_high: Vec<T>,
    ) -> Self {
        assert_eq!(f_coeff_low.len(), f_coeff_high.len());
        assert_eq!(b_coeff_low.len(), b_coeff_high.len());
        assert_eq!(f_coeff_low.len(), b_coeff_low.len());

        Self {
            f_coeff_low,
            f_coeff_high,
            b_coeff_low,
            b_coeff_high,
        }
    }
}

impl<T> Filter<T> for GenericFilter<T>
where
    T: Zero + Add<Output = T> + Mul<Output = T> + Clone + Sync,
{
    fn forwards(&self, input: &Lane<'_, T>, low: &mut [T], high: &mut [T]) {
        for (i, (low, high)) in low.iter_mut().zip(high).enumerate() {
            *low = T::zero();
            *high = T::zero();

            for (j, (lcoff, hcoff)) in self.f_coeff_low.iter().zip(&self.f_coeff_high).enumerate() {
                let idx = ((2 * i) + j).min(input.len() - 1);
                *low = low.clone() + (lcoff.clone() * input[idx].clone());
                *high = high.clone() + (hcoff.clone() * input[idx].clone());
            }
        }
    }

    fn backwards(&self, output: &mut LaneMut<'_, T>, low: &[T], high: &[T]) {
        for x in output.iter_mut() {
            *x = T::zero();
        }

        for (i, (low, high)) in low.iter().zip(high).enumerate() {
            for (j, (lcoff, hcoff)) in self.b_coeff_low.iter().zip(&self.b_coeff_high).enumerate() {
                let idx = ((2 * i) + j).min(output.len() - 1);

                let low = lcoff.clone() * low.clone();
                let high = hcoff.clone() * high.clone();
                let x = output[idx].clone() + low + high;
                output[idx] = x;
            }
        }
    }
}

impl<T> ToGenericFilter<T> for GenericFilter<T>
where
    T: Clone,
{
    fn to_generic(&self) -> GenericFilter<T> {
        self.clone()
    }
}

impl<T> Serializable for GenericFilter<T>
where
    T: Serializable,
{
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.f_coeff_low.serialize(stream);
        self.f_coeff_high.serialize(stream);
        self.b_coeff_low.serialize(stream);
        self.b_coeff_high.serialize(stream);
    }
}

impl<T> Deserializable for GenericFilter<T>
where
    T: Deserializable,
{
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let f_coeff_low = Deserializable::deserialize(stream);
        let f_coeff_high = Deserializable::deserialize(stream);
        let b_coeff_low = Deserializable::deserialize(stream);
        let b_coeff_high = Deserializable::deserialize(stream);

        Self {
            f_coeff_low,
            f_coeff_high,
            b_coeff_low,
            b_coeff_high,
        }
    }
}

/// Filter implementing an Haar wavelet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HaarWavelet;

impl<T> Filter<T> for HaarWavelet
where
    T: PartialOrd + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Clone + FloatConst,
{
    fn forwards(&self, input: &Lane<'_, T>, low: &mut [T], high: &mut [T]) {
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

    fn backwards(&self, output: &mut LaneMut<'_, T>, low: &[T], high: &[T]) {
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

impl<T> ToGenericFilter<T> for HaarWavelet
where
    T: FloatConst + Neg<Output = T>,
{
    fn to_generic(&self) -> GenericFilter<T> {
        GenericFilter::new(
            vec![T::FRAC_1_SQRT_2(), T::FRAC_1_SQRT_2()],
            vec![-T::FRAC_1_SQRT_2(), T::FRAC_1_SQRT_2()],
            vec![T::FRAC_1_SQRT_2(), T::FRAC_1_SQRT_2()],
            vec![-T::FRAC_1_SQRT_2(), T::FRAC_1_SQRT_2()],
        )
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
    T: PartialOrd + Add<Output = T> + Sub<Output = T> + Average<Output = T> + FloatConst + Clone,
{
    fn forwards(&self, input: &Lane<'_, T>, low: &mut [T], high: &mut [T]) {
        for (i, (low, high)) in low.iter_mut().zip(high).enumerate() {
            let idx_left = (2 * i).min(input.len() - 1);
            let idx_right = ((2 * i) + 1).min(input.len() - 1);

            let average = input[idx_right].clone().avg(input[idx_left].clone());
            let diff = input[idx_left].clone() - average.clone();

            *low = average;
            *high = diff;
        }
    }

    fn backwards(&self, output: &mut LaneMut<'_, T>, low: &[T], high: &[T]) {
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

impl<T> ToGenericFilter<T> for AverageFilter
where
    T: NumCast + Float,
{
    fn to_generic(&self) -> GenericFilter<T> {
        GenericFilter::new(
            vec![NumCast::from(0.5).unwrap(), NumCast::from(0.5).unwrap()],
            vec![NumCast::from(0.5).unwrap(), NumCast::from(-0.5).unwrap()],
            vec![NumCast::from(1).unwrap(), NumCast::from(1).unwrap()],
            vec![NumCast::from(1).unwrap(), NumCast::from(-1).unwrap()],
        )
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
    for mut input in input.lanes_mut(dim) {
        let scratch = &mut scratch[..input.len()];
        let (low, high) = scratch.split_at_mut(scratch.len() / 2);
        wavelet.forwards(&input.as_lane(), low, high);

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
    for mut output in output.lanes_mut(dim) {
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
    for mut output in output.lanes_mut(dim) {
        let len = output.len() / 2;

        for i in (0..len).rev() {
            let elem = output[i].clone();
            output[2 * i] = elem.clone();
            output[(2 * i) + 1] = elem;
        }
    }
}
