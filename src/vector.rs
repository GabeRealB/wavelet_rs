//! Vector data type.

use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

use num_traits::{Float, FloatConst, Num, NumCast, One, ToPrimitive, Zero};

use crate::{filter::Average, transformations::Lerp};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Vector<T, const N: usize>([T; N]);

impl<T, const N: usize> Vector<T, N> {
    /// Constructs a new `Vector`.
    #[inline]
    pub const fn new(val: [T; N]) -> Self {
        Vector(val)
    }

    /// Deconstructs the `Vector` into an array.
    #[inline]
    pub fn into_array(self) -> [T; N] {
        self.0
    }
}

impl<T, const N: usize> AsRef<[T; N]> for Vector<T, N> {
    #[inline]
    fn as_ref(&self) -> &[T; N] {
        &self.0
    }
}

impl<T, const N: usize> AsMut<[T; N]> for Vector<T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; N] {
        &mut self.0
    }
}

impl<T: Num + Clone, const N: usize> Num for Vector<T, N> {
    type FromStrRadixErr = T::FromStrRadixErr;

    #[inline]
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let mut uninit: MaybeUninit<[T; N]> = MaybeUninit::uninit();
        let uninit_ptr: *mut T = uninit.as_mut_ptr().cast();

        let t = T::from_str_radix(str, radix)?;
        for i in 0..N {
            unsafe { uninit_ptr.add(i).write(t.clone()) }
        }

        unsafe { Ok(Vector(uninit.assume_init())) }
    }
}

impl<T: Float + Clone, const N: usize> Float for Vector<T, N> {
    #[inline]
    fn nan() -> Self {
        Vector([(); N].map(|_| T::nan()))
    }

    #[inline]
    fn infinity() -> Self {
        Vector([(); N].map(|_| T::infinity()))
    }

    #[inline]
    fn neg_infinity() -> Self {
        Vector([(); N].map(|_| T::neg_infinity()))
    }

    #[inline]
    fn neg_zero() -> Self {
        Vector([(); N].map(|_| T::neg_zero()))
    }

    #[inline]
    fn min_value() -> Self {
        Vector([(); N].map(|_| T::min_value()))
    }

    #[inline]
    fn min_positive_value() -> Self {
        Vector([(); N].map(|_| T::min_positive_value()))
    }

    #[inline]
    fn max_value() -> Self {
        Vector([(); N].map(|_| T::max_value()))
    }

    #[inline]
    fn is_nan(self) -> bool {
        self.0.iter().any(|n| n.is_nan())
    }

    #[inline]
    fn is_infinite(self) -> bool {
        self.0.iter().any(|n| n.is_infinite())
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.0.iter().all(|n| n.is_infinite())
    }

    #[inline]
    fn is_normal(self) -> bool {
        self.0.iter().all(|n| n.is_normal())
    }

    #[inline]
    fn classify(self) -> std::num::FpCategory {
        unimplemented!()
    }

    #[inline]
    fn floor(self) -> Self {
        Vector(self.0.map(|n| n.floor()))
    }

    #[inline]
    fn ceil(self) -> Self {
        Vector(self.0.map(|n| n.ceil()))
    }

    #[inline]
    fn round(self) -> Self {
        Vector(self.0.map(|n| n.round()))
    }

    #[inline]
    fn trunc(self) -> Self {
        Vector(self.0.map(|n| n.trunc()))
    }

    #[inline]
    fn fract(self) -> Self {
        Vector(self.0.map(|n| n.fract()))
    }

    #[inline]
    fn abs(self) -> Self {
        Vector(self.0.map(|n| n.abs()))
    }

    #[inline]
    fn signum(self) -> Self {
        Vector(self.0.map(|n| n.signum()))
    }

    #[inline]
    fn is_sign_positive(self) -> bool {
        self.0.iter().all(|n| n.is_sign_positive())
    }

    #[inline]
    fn is_sign_negative(self) -> bool {
        self.0.iter().all(|n| n.is_sign_negative())
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Vector(zip_array(zip_array(self.0, a.0), b.0).map(|((s, a), b)| s.mul_add(a, b)))
    }

    #[inline]
    fn recip(self) -> Self {
        Vector(self.0.map(|n| n.recip()))
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        Vector(self.0.map(|x| x.powi(n)))
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        Vector(zip_array(self.0, n.0).map(|(l, r)| l.powf(r)))
    }

    #[inline]
    fn sqrt(self) -> Self {
        Vector(self.0.map(|n| n.sqrt()))
    }

    #[inline]
    fn exp(self) -> Self {
        Vector(self.0.map(|n| n.exp()))
    }

    #[inline]
    fn exp2(self) -> Self {
        Vector(self.0.map(|n| n.exp2()))
    }

    #[inline]
    fn ln(self) -> Self {
        Vector(self.0.map(|n| n.ln()))
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        Vector(zip_array(self.0, base.0).map(|(l, r)| l.log(r)))
    }

    #[inline]
    fn log2(self) -> Self {
        Vector(self.0.map(|n| n.log2()))
    }

    #[inline]
    fn log10(self) -> Self {
        Vector(self.0.map(|n| n.log10()))
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        Vector(zip_array(self.0, other.0).map(|(l, r)| l.max(r)))
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        Vector(zip_array(self.0, other.0).map(|(l, r)| l.min(r)))
    }

    #[inline]
    fn abs_sub(self, other: Self) -> Self {
        Vector(zip_array(self.0, other.0).map(|(l, r)| l.abs_sub(r)))
    }

    #[inline]
    fn cbrt(self) -> Self {
        Vector(self.0.map(|n| n.cbrt()))
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        Vector(zip_array(self.0, other.0).map(|(l, r)| l.hypot(r)))
    }

    #[inline]
    fn sin(self) -> Self {
        Vector(self.0.map(|n| n.sin()))
    }

    #[inline]
    fn cos(self) -> Self {
        Vector(self.0.map(|n| n.cos()))
    }

    #[inline]
    fn tan(self) -> Self {
        Vector(self.0.map(|n| n.tan()))
    }

    #[inline]
    fn asin(self) -> Self {
        Vector(self.0.map(|n| n.asin()))
    }

    #[inline]
    fn acos(self) -> Self {
        Vector(self.0.map(|n| n.acos()))
    }

    #[inline]
    fn atan(self) -> Self {
        Vector(self.0.map(|n| n.atan()))
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        Vector(zip_array(self.0, other.0).map(|(l, r)| l.atan2(r)))
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = unzip_array(self.0.map(|n| n.sin_cos()));
        (Vector(sin), Vector(cos))
    }

    #[inline]
    fn exp_m1(self) -> Self {
        Vector(self.0.map(|n| n.exp_m1()))
    }

    #[inline]
    fn ln_1p(self) -> Self {
        Vector(self.0.map(|n| n.ln_1p()))
    }

    #[inline]
    fn sinh(self) -> Self {
        Vector(self.0.map(|n| n.sinh()))
    }

    #[inline]
    fn cosh(self) -> Self {
        Vector(self.0.map(|n| n.cosh()))
    }

    #[inline]
    fn tanh(self) -> Self {
        Vector(self.0.map(|n| n.tanh()))
    }

    #[inline]
    fn asinh(self) -> Self {
        Vector(self.0.map(|n| n.asinh()))
    }

    #[inline]
    fn acosh(self) -> Self {
        Vector(self.0.map(|n| n.acosh()))
    }

    #[inline]
    fn atanh(self) -> Self {
        Vector(self.0.map(|n| n.atanh()))
    }

    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        unimplemented!()
    }
}

impl<T: FloatConst, const N: usize> FloatConst for Vector<T, N> {
    fn E() -> Self {
        Vector([(); N].map(|_| T::E()))
    }

    fn FRAC_1_PI() -> Self {
        Vector([(); N].map(|_| T::FRAC_1_PI()))
    }

    fn FRAC_1_SQRT_2() -> Self {
        Vector([(); N].map(|_| T::FRAC_1_SQRT_2()))
    }

    fn FRAC_2_PI() -> Self {
        Vector([(); N].map(|_| T::FRAC_2_PI()))
    }

    fn FRAC_2_SQRT_PI() -> Self {
        Vector([(); N].map(|_| T::FRAC_2_SQRT_PI()))
    }

    fn FRAC_PI_2() -> Self {
        Vector([(); N].map(|_| T::FRAC_PI_2()))
    }

    fn FRAC_PI_3() -> Self {
        Vector([(); N].map(|_| T::FRAC_PI_3()))
    }

    fn FRAC_PI_4() -> Self {
        Vector([(); N].map(|_| T::FRAC_PI_4()))
    }

    fn FRAC_PI_6() -> Self {
        Vector([(); N].map(|_| T::FRAC_PI_6()))
    }

    fn FRAC_PI_8() -> Self {
        Vector([(); N].map(|_| T::FRAC_PI_8()))
    }

    fn LN_10() -> Self {
        Vector([(); N].map(|_| T::LN_10()))
    }

    fn LN_2() -> Self {
        Vector([(); N].map(|_| T::LN_2()))
    }

    fn LOG10_E() -> Self {
        Vector([(); N].map(|_| T::LOG10_E()))
    }

    fn LOG2_E() -> Self {
        Vector([(); N].map(|_| T::LOG2_E()))
    }

    fn PI() -> Self {
        Vector([(); N].map(|_| T::PI()))
    }

    fn SQRT_2() -> Self {
        Vector([(); N].map(|_| T::SQRT_2()))
    }
}

impl<T: NumCast + Clone, const N: usize> NumCast for Vector<T, N> {
    #[inline]
    fn from<U: num_traits::ToPrimitive>(n: U) -> Option<Self> {
        let t = T::from(n)?;
        Some(Vector([(); N].map(|_| t.clone())))
    }
}

impl<T: ToPrimitive, const N: usize> ToPrimitive for Vector<T, N> {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        None
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        None
    }
}

impl<T: Zero, const N: usize> Zero for Vector<T, N> {
    #[inline]
    fn zero() -> Self {
        Vector([(); N].map(|_| T::zero()))
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.iter().all(|n| n.is_zero())
    }
}

impl<T: One, const N: usize> One for Vector<T, N> {
    #[inline]
    fn one() -> Self {
        Vector([(); N].map(|_| T::one()))
    }
}

impl<T: Add, const N: usize> Add for Vector<T, N> {
    type Output = Vector<T::Output, N>;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Vector(zip_array(self.0, rhs.0).map(|(l, r)| l + r))
    }
}

impl<T: Sub, const N: usize> Sub for Vector<T, N> {
    type Output = Vector<T::Output, N>;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Vector(zip_array(self.0, rhs.0).map(|(l, r)| l - r))
    }
}

impl<T: Mul, const N: usize> Mul for Vector<T, N> {
    type Output = Vector<T::Output, N>;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Vector(zip_array(self.0, rhs.0).map(|(l, r)| l * r))
    }
}

impl<T: Div, const N: usize> Div for Vector<T, N> {
    type Output = Vector<T::Output, N>;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Vector(zip_array(self.0, rhs.0).map(|(l, r)| l / r))
    }
}

impl<T: Rem, const N: usize> Rem for Vector<T, N> {
    type Output = Vector<T::Output, N>;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        Vector(zip_array(self.0, rhs.0).map(|(l, r)| l % r))
    }
}

impl<T: Neg, const N: usize> Neg for Vector<T, N> {
    type Output = Vector<T::Output, N>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector(self.0.map(|v| v.neg()))
    }
}

impl<T: Lerp, const N: usize> Lerp for Vector<T, N> {
    type Output = Vector<T::Output, N>;

    #[inline]
    fn lerp(self, other: Self, t: f32) -> Self {
        Vector(zip_array(self.0, other.0).map(|(l, r)| l.lerp(r, t)))
    }
}

impl<T: Average<Output = T>, const N: usize> Average for Vector<T, N> {
    type Output = Self;

    #[inline]
    fn avg(self, rhs: Self) -> Self::Output {
        Vector(zip_array(self.0, rhs.0).map(|(l, r)| l.avg(r)))
    }
}

#[inline]
fn zip_array<T, U, const N: usize>(lhs: [T; N], rhs: [U; N]) -> [(T, U); N] {
    let lhs = ManuallyDrop::new(lhs);
    let rhs = ManuallyDrop::new(rhs);

    let mut uninit: MaybeUninit<[(T, U); N]> = MaybeUninit::uninit();
    let uninit_ptr: *mut (T, U) = uninit.as_mut_ptr().cast();
    for i in 0..N {
        let l: *const T = &lhs[i];
        let r: *const U = &rhs[i];

        unsafe {
            let l = l.read();
            let r = r.read();
            uninit_ptr.add(i).write((l, r))
        }
    }

    unsafe { uninit.assume_init() }
}

#[inline]
fn unzip_array<T, U, const N: usize>(array: [(T, U); N]) -> ([T; N], [U; N]) {
    let array = ManuallyDrop::new(array);

    let mut uninit_l: MaybeUninit<[T; N]> = MaybeUninit::uninit();
    let mut uninit_r: MaybeUninit<[U; N]> = MaybeUninit::uninit();
    let l_ptr: *mut T = uninit_l.as_mut_ptr().cast();
    let r_ptr: *mut U = uninit_r.as_mut_ptr().cast();
    for i in 0..N {
        let l: *const T = &array[i].0;
        let r: *const U = &array[i].1;

        unsafe {
            l_ptr.add(i).write(l.read());
            r_ptr.add(i).write(r.read());
        }
    }

    unsafe { (uninit_l.assume_init(), uninit_r.assume_init()) }
}
