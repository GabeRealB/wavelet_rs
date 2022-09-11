use num_traits::Num;

use crate::stream::{Deserializable, Serializable};

use super::{Backwards, Forwards, OneWayTransform};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Identity;

impl<N: Num + Copy> OneWayTransform<Forwards, N> for Identity {
    type Cfg<'a> = Identity;

    #[inline(always)]
    fn apply(
        &self,
        input: crate::volume::VolumeBlock<N>,
        _cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<N> {
        input
    }
}

impl<N: Num + Copy> OneWayTransform<Backwards, N> for Identity {
    type Cfg<'a> = Identity;

    #[inline(always)]
    fn apply(
        &self,
        input: crate::volume::VolumeBlock<N>,
        _cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<N> {
        input
    }
}

impl Serializable for Identity {
    fn serialize(self, _stream: &mut crate::stream::SerializeStream) {}
}

impl Deserializable for Identity {
    fn deserialize(_stream: &mut crate::stream::DeserializeStream<'_>) -> Self {
        Self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

impl<T, N: Num + Copy> OneWayTransform<Forwards, N> for Reverse<T>
where
    T: OneWayTransform<Backwards, N>,
{
    type Cfg<'a> = T::Cfg<'a>;

    #[inline(always)]
    fn apply(
        &self,
        input: crate::volume::VolumeBlock<N>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<N> {
        self.0.apply(input, cfg)
    }
}

impl<T, N: Num + Copy> OneWayTransform<Backwards, N> for Reverse<T>
where
    T: OneWayTransform<Forwards, N>,
{
    type Cfg<'a> = T::Cfg<'a>;

    #[inline(always)]
    fn apply(
        &self,
        input: crate::volume::VolumeBlock<N>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<N> {
        self.0.apply(input, cfg)
    }
}

impl<T: Serializable> Serializable for Reverse<T> {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        T::name().serialize(stream);
        self.0.serialize(stream);
    }
}

impl<T: Deserializable> Deserializable for Reverse<T> {
    fn deserialize(stream: &mut crate::stream::DeserializeStream<'_>) -> Self {
        let t_ty: String = Deserializable::deserialize(stream);
        assert_eq!(t_ty, T::name());

        let elem = Deserializable::deserialize(stream);
        Self(elem)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

    /// Combines two operations.
    #[inline]
    pub fn combine<'a>(t: T, u: U) -> Chain<T, U>
    where
        T: 'a,
        U: 'a,
    {
        Chain(t, u)
    }
}

impl<T, U> From<(T, U)> for Chain<T, U> {
    #[inline]
    fn from(t: (T, U)) -> Self {
        Chain(t.0, t.1)
    }
}

impl<N, T, U> OneWayTransform<Forwards, N> for Chain<T, U>
where
    N: Num + Copy,
    T: OneWayTransform<Forwards, N>,
    U: OneWayTransform<Forwards, N>,
{
    type Cfg<'a> = Chain<T::Cfg<'a>, U::Cfg<'a>>;

    fn apply(
        &self,
        input: crate::volume::VolumeBlock<N>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<N> {
        let tmp = self.0.apply(input, cfg.0);
        self.1.apply(tmp, cfg.1)
    }
}

impl<N, T, U> OneWayTransform<Backwards, N> for Chain<T, U>
where
    N: Num + Copy,
    T: OneWayTransform<Backwards, N>,
    U: OneWayTransform<Backwards, N>,
{
    type Cfg<'a> = Chain<T::Cfg<'a>, U::Cfg<'a>>;

    fn apply(
        &self,
        input: crate::volume::VolumeBlock<N>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<N> {
        let tmp = self.1.apply(input, cfg.1);
        self.0.apply(tmp, cfg.0)
    }
}

impl<T: Serializable, U: Serializable> Serializable for Chain<T, U> {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        T::name().serialize(stream);
        U::name().serialize(stream);
        self.0.serialize(stream);
        self.1.serialize(stream);
    }
}

impl<T: Deserializable, U: Deserializable> Deserializable for Chain<T, U> {
    fn deserialize(stream: &mut crate::stream::DeserializeStream<'_>) -> Self {
        let t_ty: String = Deserializable::deserialize(stream);
        let u_ty: String = Deserializable::deserialize(stream);
        assert_eq!(t_ty, T::name());
        assert_eq!(u_ty, U::name());

        let elem_0 = Deserializable::deserialize(stream);
        let elem_1 = Deserializable::deserialize(stream);
        Self(elem_0, elem_1)
    }
}
