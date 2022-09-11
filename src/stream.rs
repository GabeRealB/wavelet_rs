use std::{
    borrow::Borrow,
    collections::BTreeMap,
    io::{Read, Write},
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

use crate::vector::Vector;

pub trait Named {
    fn name() -> &'static str
    where
        Self: Sized;

    fn get_name(&self) -> &'static str;
}

impl<T: ?Sized> Named for T {
    fn name() -> &'static str
    where
        Self: Sized,
    {
        std::any::type_name::<T>()
    }

    fn get_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }
}

pub trait Serializable: Named {
    fn serialize(self, stream: &mut SerializeStream);
}

pub trait Deserializable: Named {
    fn deserialize(stream: &mut DeserializeStream<'_>) -> Self;
}

#[derive(Debug, Default)]
pub struct SerializeStream {
    bytes: Vec<u8>,
}

impl SerializeStream {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    pub fn write(&self, x: &mut impl Write) -> std::io::Result<()> {
        x.write_all(&self.bytes)
    }
}

impl Write for SerializeStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct DeserializeStream<'a> {
    idx: usize,
    bytes: &'a [u8],
}

impl<'a> DeserializeStream<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { idx: 0, bytes: buf }
    }
}

impl<'a> Read for DeserializeStream<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let read = Read::read(&mut &self.bytes[self.idx..], buf)?;
        self.idx += read;
        Ok(read)
    }
}

macro_rules! impl_ser {
    ($($T:ty, $S:literal);*) => {
        $(
            impl Serializable for $T {
                #[inline]
                fn serialize(self, stream: &mut SerializeStream) {
                    let bytes = self.to_le_bytes();
                    stream.write_all(&bytes).unwrap();
                }
            }

            impl Deserializable for $T {
                #[inline]
                fn deserialize(stream: &mut DeserializeStream<'_>) -> Self {
                    let mut bytes = [0; $S];
                    stream.read_exact(&mut bytes).unwrap();
                    <$T>::from_le_bytes(bytes)
                }
            }
        )*
    };
}

impl_ser! {
    u8, 1;
    u16, 2;
    u32, 4;
    u64, 8;
    usize, 8;
    i8, 1;
    i16, 2;
    i32, 4;
    i64, 8;
    isize, 8;
    f32, 4;
    f64, 8
}

impl Serializable for bool {
    #[inline]
    fn serialize(self, stream: &mut SerializeStream) {
        (self as u8).serialize(stream)
    }
}

impl Deserializable for bool {
    #[inline]
    fn deserialize(stream: &mut DeserializeStream<'_>) -> Self {
        let val: u8 = Deserializable::deserialize(stream);
        val == 1
    }
}

impl Serializable for &str {
    #[inline]
    fn serialize(self, stream: &mut SerializeStream) {
        self.len().serialize(stream);
        stream.write_all(self.as_bytes()).unwrap();
    }
}

impl Serializable for &mut str {
    #[inline]
    fn serialize(self, stream: &mut SerializeStream) {
        self.len().serialize(stream);
        stream.write_all(self.as_bytes()).unwrap();
    }
}

impl<T: Serializable + Clone> Serializable for &[T] {
    #[inline]
    fn serialize(self, stream: &mut SerializeStream) {
        self.len().serialize(stream);
        for x in self {
            x.clone().serialize(stream)
        }
    }
}
impl<T: Serializable + Clone> Serializable for &mut [T] {
    #[inline]
    fn serialize(self, stream: &mut SerializeStream) {
        self.len().serialize(stream);
        for x in self {
            x.clone().serialize(stream)
        }
    }
}

impl Serializable for String {
    #[inline]
    fn serialize(self, stream: &mut SerializeStream) {
        self.as_str().serialize(stream)
    }
}

impl Deserializable for String {
    #[inline]
    fn deserialize(stream: &mut DeserializeStream<'_>) -> Self {
        let len: usize = Deserializable::deserialize(stream);
        let mut str = String::with_capacity(len);
        for _ in 0..len {
            str.push('\0');
        }

        let mut bytes_read = 0;
        while bytes_read != len {
            let buf = unsafe { &mut str.as_bytes_mut()[bytes_read..] };
            bytes_read += stream.read(buf).unwrap();
        }
        str
    }
}

impl<T: Serializable, const N: usize> Serializable for Vector<T, N> {
    #[inline]
    fn serialize(self, stream: &mut SerializeStream) {
        for x in self.into_array() {
            x.serialize(stream);
        }
    }
}

impl<T: Deserializable, const N: usize> Deserializable for Vector<T, N> {
    #[inline]
    fn deserialize(stream: &mut DeserializeStream<'_>) -> Self {
        let x = [(); N].map(|_| Deserializable::deserialize(stream));
        Vector::new(x)
    }
}

impl<T: Serializable> Serializable for Vec<T> {
    #[inline]
    fn serialize(self, stream: &mut SerializeStream) {
        self.len().serialize(stream);
        for x in self {
            x.serialize(stream);
        }
    }
}

impl<T: Deserializable> Deserializable for Vec<T> {
    #[inline]
    fn deserialize(stream: &mut DeserializeStream<'_>) -> Self {
        let len: usize = Deserializable::deserialize(stream);
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(Deserializable::deserialize(stream));
        }

        vec
    }
}

impl<T: Serializable> Serializable for Range<T> {
    #[inline]
    fn serialize(self, stream: &mut SerializeStream) {
        self.start.serialize(stream);
        self.end.serialize(stream);
    }
}

impl<T: Deserializable> Deserializable for Range<T> {
    #[inline]
    fn deserialize(stream: &mut DeserializeStream<'_>) -> Self {
        let start = Deserializable::deserialize(stream);
        let end = Deserializable::deserialize(stream);

        Self { start, end }
    }
}

impl<T: Serializable> Serializable for RangeFrom<T> {
    #[inline]
    fn serialize(self, stream: &mut SerializeStream) {
        self.start.serialize(stream);
    }
}

impl<T: Deserializable> Deserializable for RangeFrom<T> {
    #[inline]
    fn deserialize(stream: &mut DeserializeStream<'_>) -> Self {
        let start = Deserializable::deserialize(stream);

        Self { start }
    }
}

impl Serializable for RangeFull {
    #[inline]
    fn serialize(self, _stream: &mut SerializeStream) {}
}

impl Deserializable for RangeFull {
    #[inline]
    fn deserialize(_stream: &mut DeserializeStream<'_>) -> Self {
        Self
    }
}

impl<T: Serializable> Serializable for RangeInclusive<T> {
    #[inline]
    fn serialize(self, stream: &mut SerializeStream) {
        let (start, end) = self.into_inner();

        start.serialize(stream);
        end.serialize(stream);
    }
}

impl<T: Deserializable> Deserializable for RangeInclusive<T> {
    #[inline]
    fn deserialize(stream: &mut DeserializeStream<'_>) -> Self {
        let start = Deserializable::deserialize(stream);
        let end = Deserializable::deserialize(stream);

        Self::new(start, end)
    }
}

impl<T: Serializable> Serializable for RangeTo<T> {
    #[inline]
    fn serialize(self, stream: &mut SerializeStream) {
        self.end.serialize(stream);
    }
}

impl<T: Deserializable> Deserializable for RangeTo<T> {
    #[inline]
    fn deserialize(stream: &mut DeserializeStream<'_>) -> Self {
        let end = Deserializable::deserialize(stream);

        Self { end }
    }
}

impl<T: Serializable> Serializable for RangeToInclusive<T> {
    #[inline]
    fn serialize(self, stream: &mut SerializeStream) {
        self.end.serialize(stream);
    }
}

impl<T: Deserializable> Deserializable for RangeToInclusive<T> {
    #[inline]
    fn deserialize(stream: &mut DeserializeStream<'_>) -> Self {
        let end = Deserializable::deserialize(stream);

        Self { end }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct AnyMapItem {
    r#type: String,
    data: Box<[u8]>,
}

impl Serializable for AnyMapItem {
    fn serialize(self, stream: &mut SerializeStream) {
        self.r#type.serialize(stream);
        self.data.len().serialize(stream);
        stream.write_all(&self.data).unwrap();
    }
}

impl Deserializable for AnyMapItem {
    fn deserialize(stream: &mut DeserializeStream<'_>) -> Self {
        let r#type: String = Deserializable::deserialize(stream);
        let data_len: usize = Deserializable::deserialize(stream);
        let mut data = vec![0u8; data_len].into_boxed_slice();

        let mut bytes_read = 0;
        while bytes_read != data_len {
            let buf = &mut data[bytes_read..];
            bytes_read += stream.read(buf).unwrap();
        }

        Self { r#type, data }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AnyMap {
    map: BTreeMap<String, AnyMapItem>,
}

impl AnyMap {
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn get<Q, T>(&self, key: &Q) -> Option<T>
    where
        String: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
        T: Deserializable,
    {
        let val = self.map.get(key)?;
        if val.r#type != T::name() {
            None
        } else {
            let mut stream = DeserializeStream::new(&val.data);
            Some(T::deserialize(&mut stream))
        }
    }

    pub fn insert<T>(&mut self, key: String, value: T) -> bool
    where
        T: Serializable,
    {
        let r#type = T::name();

        let mut stream = SerializeStream::new();
        value.serialize(&mut stream);
        let buf = stream.bytes.into_boxed_slice();

        let value = AnyMapItem {
            r#type: r#type.into(),
            data: buf,
        };
        self.map.insert(key, value).is_some()
    }
}

impl Serializable for AnyMap {
    fn serialize(self, stream: &mut SerializeStream) {
        self.map.len().serialize(stream);
        for (k, v) in self.map {
            k.serialize(stream);
            v.serialize(stream);
        }
    }
}

impl Deserializable for AnyMap {
    fn deserialize(stream: &mut DeserializeStream<'_>) -> Self {
        let map_len: usize = Deserializable::deserialize(stream);
        let mut map = BTreeMap::new();
        for _ in 0..map_len {
            let key = Deserializable::deserialize(stream);
            let value = Deserializable::deserialize(stream);
            map.insert(key, value);
        }

        Self { map }
    }
}
