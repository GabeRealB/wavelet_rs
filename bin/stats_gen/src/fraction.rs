use std::{
    ops::{Add, Div, Mul, Sub},
    str::FromStr,
};

use fraction::{ToPrimitive, Zero};
use num_traits::NumCast;
use wavelet_rs::{
    filter::Average,
    stream::{Deserializable, Serializable},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fraction {
    pub frac: fraction::BigFraction,
}

impl Add for Fraction {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Fraction {
            frac: self.frac + rhs.frac,
        }
    }
}

impl Sub for Fraction {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Fraction {
            frac: self.frac - rhs.frac,
        }
    }
}

impl Mul for Fraction {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Fraction {
            frac: self.frac * rhs.frac,
        }
    }
}

impl Div for Fraction {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Fraction {
            frac: self.frac / rhs.frac,
        }
    }
}

impl<T> Add<T> for Fraction
where
    fraction::BigFraction: Add<T, Output = fraction::BigFraction>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Fraction {
            frac: self.frac + rhs,
        }
    }
}

impl<T> Sub<T> for Fraction
where
    fraction::BigFraction: Sub<T, Output = fraction::BigFraction>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Fraction {
            frac: self.frac - rhs,
        }
    }
}

impl<T> Mul<T> for Fraction
where
    fraction::BigFraction: Mul<T, Output = fraction::BigFraction>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Fraction {
            frac: self.frac * rhs,
        }
    }
}

impl<T> Div<T> for Fraction
where
    fraction::BigFraction: Div<T, Output = fraction::BigFraction>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Fraction {
            frac: self.frac / rhs,
        }
    }
}

impl Zero for Fraction {
    fn zero() -> Self {
        0.into()
    }

    fn is_zero(&self) -> bool {
        self.frac.is_zero()
    }
}

impl ToPrimitive for Fraction {
    fn to_i64(&self) -> Option<i64> {
        self.frac.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.frac.to_u64()
    }
}

impl Average for Fraction {
    type Output = Self;

    fn avg(self, rhs: Self) -> Self::Output {
        (self + rhs) / 2
    }
}

impl NumCast for Fraction {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        let n: f64 = n.to_f64()?;
        Some(Fraction { frac: n.into() })
    }
}

impl Serializable for Fraction {
    fn serialize(self, stream: &mut wavelet_rs::stream::SerializeStream) {
        self.frac.to_string().serialize(stream);
    }
}

impl Deserializable for Fraction {
    fn deserialize(stream: &mut wavelet_rs::stream::DeserializeStreamRef<'_>) -> Self {
        let str: String = String::deserialize(stream);

        Fraction {
            frac: fraction::BigFraction::from_str(&str).unwrap(),
        }
    }
}

impl<T> From<T> for Fraction
where
    fraction::BigFraction: From<T>,
{
    fn from(value: T) -> Self {
        Fraction { frac: value.into() }
    }
}

impl From<Fraction> for f64 {
    fn from(value: Fraction) -> Self {
        value.frac.to_f64().unwrap()
    }
}
