use super::{Backwards, Forwards, OneWayTransform};
use crate::stream::{Deserializable, Serializable};

/// Identity transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Identity;

impl<T> OneWayTransform<Forwards, T> for Identity {
    type Cfg<'a> = Identity;

    #[inline(always)]
    fn apply(
        &self,
        input: crate::volume::VolumeBlock<T>,
        _cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<T> {
        input
    }
}

impl<T> OneWayTransform<Backwards, T> for Identity {
    type Cfg<'a> = Identity;

    #[inline(always)]
    fn apply(
        &self,
        input: crate::volume::VolumeBlock<T>,
        _cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<T> {
        input
    }
}

impl Serializable for Identity {
    fn serialize(self, _stream: &mut crate::stream::SerializeStream) {}
}

impl Deserializable for Identity {
    fn deserialize(_stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        Self
    }
}

/// Reverses the direction of the transformation.
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

impl<T, U> OneWayTransform<Forwards, U> for Reverse<T>
where
    T: OneWayTransform<Backwards, U>,
{
    type Cfg<'a> = T::Cfg<'a>;

    #[inline(always)]
    fn apply(
        &self,
        input: crate::volume::VolumeBlock<U>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<U> {
        self.0.apply(input, cfg)
    }
}

impl<T, U> OneWayTransform<Backwards, U> for Reverse<T>
where
    T: OneWayTransform<Forwards, U>,
{
    type Cfg<'a> = T::Cfg<'a>;

    #[inline(always)]
    fn apply(
        &self,
        input: crate::volume::VolumeBlock<U>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<U> {
        self.0.apply(input, cfg)
    }
}

impl<T: Serializable> Serializable for Reverse<T> {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        std::any::type_name::<T>().serialize(stream);
        self.0.serialize(stream);
    }
}

impl<T: Deserializable> Deserializable for Reverse<T> {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let t_ty: String = Deserializable::deserialize(stream);
        assert_eq!(t_ty, std::any::type_name::<T>());

        let elem = Deserializable::deserialize(stream);
        Self(elem)
    }
}

/// Chain of two transformations.
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

impl<T, U, V> OneWayTransform<Forwards, V> for Chain<T, U>
where
    T: OneWayTransform<Forwards, V>,
    U: OneWayTransform<Forwards, V>,
{
    type Cfg<'a> = Chain<T::Cfg<'a>, U::Cfg<'a>>;

    fn apply(
        &self,
        input: crate::volume::VolumeBlock<V>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<V> {
        let tmp = self.0.apply(input, cfg.0);
        self.1.apply(tmp, cfg.1)
    }
}

impl<T, U, V> OneWayTransform<Backwards, V> for Chain<T, U>
where
    T: OneWayTransform<Backwards, V>,
    U: OneWayTransform<Backwards, V>,
{
    type Cfg<'a> = Chain<T::Cfg<'a>, U::Cfg<'a>>;

    fn apply(
        &self,
        input: crate::volume::VolumeBlock<V>,
        cfg: Self::Cfg<'_>,
    ) -> crate::volume::VolumeBlock<V> {
        let tmp = self.1.apply(input, cfg.1);
        self.0.apply(tmp, cfg.0)
    }
}

impl<T: Serializable, U: Serializable> Serializable for Chain<T, U> {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        std::any::type_name::<T>().serialize(stream);
        std::any::type_name::<U>().serialize(stream);
        self.0.serialize(stream);
        self.1.serialize(stream);
    }
}

impl<T: Deserializable, U: Deserializable> Deserializable for Chain<T, U> {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let t_ty: String = Deserializable::deserialize(stream);
        let u_ty: String = Deserializable::deserialize(stream);
        assert_eq!(t_ty, std::any::type_name::<T>());
        assert_eq!(u_ty, std::any::type_name::<U>());

        let elem_0 = Deserializable::deserialize(stream);
        let elem_1 = Deserializable::deserialize(stream);
        Self(elem_0, elem_1)
    }
}
