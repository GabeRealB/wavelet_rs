use num_traits::Num;

use super::Transformation;

pub struct Identity;

impl<T: Num + Copy> Transformation<T> for Identity {
    #[inline(always)]
    fn forwards(&self, input: crate::volume::VolumeBlock<T>) -> crate::volume::VolumeBlock<T> {
        input
    }

    #[inline(always)]
    fn backwards(&self, input: crate::volume::VolumeBlock<T>) -> crate::volume::VolumeBlock<T> {
        input
    }
}

pub struct Reverse<T>(T);

impl<T> Reverse<T> {
    /// Constructs a new `Reverse`.
    #[inline]
    pub fn new(t: T) -> Self {
        Reverse(t)
    }
}

impl<T> From<T> for Reverse<T> {
    #[inline]
    fn from(t: T) -> Self {
        Self::new(t)
    }
}

impl<N, T> Transformation<N> for Reverse<T>
where
    N: Num + Copy,
    T: Transformation<N>,
{
    #[inline(always)]
    fn forwards(&self, input: crate::volume::VolumeBlock<N>) -> crate::volume::VolumeBlock<N> {
        self.0.backwards(input)
    }

    #[inline(always)]
    fn backwards(&self, input: crate::volume::VolumeBlock<N>) -> crate::volume::VolumeBlock<N> {
        self.0.forwards(input)
    }
}

pub struct Chain<T, U>(T, U);

impl<T, U> Chain<T, U> {
    /// Constructs a new `Chain`.
    #[inline]
    pub fn new(t: U) -> Chain<Identity, U> {
        Chain(Identity, t)
    }

    /// Chains another operation at the end.
    #[inline]
    pub fn chain<V>(self, t: V) -> Chain<Self, V> {
        Chain(self, t)
    }
}

impl<T, U> From<(T, U)> for Chain<T, U> {
    #[inline]
    fn from(t: (T, U)) -> Self {
        Chain(t.0, t.1)
    }
}

impl<N, T, U> Transformation<N> for Chain<T, U>
where
    N: Num + Copy,
    T: Transformation<N>,
    U: Transformation<N>,
{
    #[inline(always)]
    fn forwards(&self, input: crate::volume::VolumeBlock<N>) -> crate::volume::VolumeBlock<N> {
        let tmp = self.0.forwards(input);
        self.1.forwards(tmp)
    }

    #[inline(always)]
    fn backwards(&self, input: crate::volume::VolumeBlock<N>) -> crate::volume::VolumeBlock<N> {
        let tmp = self.1.backwards(input);
        self.0.backwards(tmp)
    }
}
