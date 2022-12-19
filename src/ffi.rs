//! C-API of the library.

use std::{
    ffi::{c_char, CStr},
    fmt::Debug,
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    ops::{Deref, DerefMut, Range},
};

use paste::paste;

use crate::{
    decoder::VolumeWaveletDecoder,
    encoder::VolumeWaveletEncoder,
    filter::{AverageFilter, HaarWavelet},
    stream::{Deserializable, Serializable},
};

#[cfg(feature = "ffi_vec")]
use crate::vector::Vector;

/// C compatible array type.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CArray<T, const N: usize>([T; N]);

impl<T, const N: usize> From<[T; N]> for CArray<T, N> {
    fn from(x: [T; N]) -> Self {
        Self(x)
    }
}

impl<T, const N: usize> From<CArray<T, N>> for [T; N] {
    fn from(x: CArray<T, N>) -> Self {
        x.0
    }
}

impl<T: Serializable, const N: usize> Serializable for CArray<T, N> {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.0.serialize(stream)
    }
}

impl<T: Deserializable, const N: usize> Deserializable for CArray<T, N> {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let arr = Deserializable::deserialize(stream);
        Self(arr)
    }
}

/// C compatible range type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CRange<T> {
    start: T,
    end: T,
}

impl<T> From<Range<T>> for CRange<T> {
    fn from(x: Range<T>) -> Self {
        Self {
            start: x.start,
            end: x.end,
        }
    }
}

impl<T> From<CRange<T>> for Range<T> {
    fn from(x: CRange<T>) -> Self {
        Range {
            start: x.start,
            end: x.end,
        }
    }
}

/// C compatible slice type.
#[repr(C)]
pub struct CSlice<'a, T> {
    ptr: *const T,
    len: usize,
    _phantom: PhantomData<&'a [T]>,
}

impl<'a, T> Deref for CSlice<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<'a, T> Copy for CSlice<'a, T> {}

impl<'a, T> Clone for CSlice<'a, T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: Debug> Debug for CSlice<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<'a, T: PartialEq> PartialEq for CSlice<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<'a, T: Eq> Eq for CSlice<'a, T> {}

impl<'a, T: PartialOrd> PartialOrd for CSlice<'a, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<'a, T: Ord> Ord for CSlice<'a, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<'a, T> From<&'a [T]> for CSlice<'a, T> {
    fn from(x: &'a [T]) -> Self {
        CSlice {
            ptr: x.as_ptr(),
            len: x.len(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, T> From<CSlice<'a, T>> for &'a [T] {
    fn from(x: CSlice<'a, T>) -> Self {
        unsafe { std::slice::from_raw_parts(x.ptr, x.len) }
    }
}

/// C compatible owned slice type.
#[repr(C)]
pub struct OwnedCSlice<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Deref for OwnedCSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> DerefMut for OwnedCSlice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Clone> Clone for OwnedCSlice<T> {
    fn clone(&self) -> Self {
        let vec: Vec<_> = (**self).into();
        vec.into_boxed_slice().into()
    }
}

impl<T: Debug> Debug for OwnedCSlice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<T: PartialEq> PartialEq for OwnedCSlice<T> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<T: Eq> Eq for OwnedCSlice<T> {}

impl<T: PartialOrd> PartialOrd for OwnedCSlice<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<T: Ord> Ord for OwnedCSlice<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T> From<Box<[T]>> for OwnedCSlice<T> {
    fn from(x: Box<[T]>) -> Self {
        let mut x = ManuallyDrop::new(x);
        Self {
            ptr: x.as_mut_ptr(),
            len: x.len(),
        }
    }
}

impl<T> From<OwnedCSlice<T>> for Box<[T]> {
    fn from(x: OwnedCSlice<T>) -> Self {
        let x = ManuallyDrop::new(x);
        unsafe { Box::from_raw(std::slice::from_raw_parts_mut(x.ptr, x.len)) }
    }
}

impl<T: Serializable> Serializable for OwnedCSlice<T> {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        let b: Box<[T]> = self.into();
        b.serialize(stream)
    }
}

impl<T: Deserializable> Deserializable for OwnedCSlice<T> {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let b = Box::deserialize(stream);
        b.into()
    }
}

impl<T> Drop for OwnedCSlice<T> {
    fn drop(&mut self) {
        unsafe { Box::from_raw(std::slice::from_raw_parts_mut(self.ptr, self.len)) };
    }
}

macro_rules! owned_c_slice_def {
    ($($T:ty);*) => {
        paste! {
            $(
                /// Allocates a new owned slice.
                #[no_mangle]
                pub extern "C" fn [<wavelet_rs_slice_ $T _new>](data: CSlice<'_, $T>) -> OwnedCSlice<$T> {
                    let data: &[$T] = data.into();
                    let vec: Vec<_> = data.into();
                    vec.into_boxed_slice().into()
                }

                /// Deallocates the owned slice.
                #[no_mangle]
                pub extern "C" fn [<wavelet_rs_slice_ $T _free>](_: OwnedCSlice<$T>) {}
            )*
        }
    };
}

owned_c_slice_def! {
    u8; u16; u32; u64; i8; i16; i32; i64; f32; f64; c_char; CString
}

/// C compatible owned string.
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CString {
    buff: OwnedCSlice<c_char>,
}

impl Serializable for CString {
    fn serialize(self, stream: &mut crate::stream::SerializeStream) {
        self.buff.serialize(stream)
    }
}

impl Deserializable for CString {
    fn deserialize(stream: &mut crate::stream::DeserializeStreamRef<'_>) -> Self {
        let buff = Deserializable::deserialize(stream);
        Self { buff }
    }
}

/// C compatible option type.
#[repr(C, i8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum COption<T> {
    /// Empty variant.
    None,
    /// Filled variant.
    Some(T),
}

impl<T> From<Option<T>> for COption<T> {
    fn from(x: Option<T>) -> Self {
        match x {
            None => Self::None,
            Some(x) => Self::Some(x),
        }
    }
}

impl<T> From<COption<T>> for Option<T> {
    fn from(x: COption<T>) -> Self {
        match x {
            COption::None => None,
            COption::Some(x) => Some(x),
        }
    }
}

/// Element types exposed by the ffi api.
#[repr(i32)]
#[allow(missing_docs)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ElemType {
    F32 = 0,

    #[cfg(feature = "ffi_vec")]
    F32Vec1 = 1,
    #[cfg(feature = "ffi_vec")]
    F32Vec2,
    #[cfg(feature = "ffi_vec")]
    F32Vec3,
    #[cfg(feature = "ffi_vec")]
    F32Vec4,

    F64 = 40,

    #[cfg(feature = "ffi_vec")]
    F64Vec1 = 41,
    #[cfg(feature = "ffi_vec")]
    F64Vec2,
    #[cfg(feature = "ffi_vec")]
    F64Vec3,
    #[cfg(feature = "ffi_vec")]
    F64Vec4,
}

impl From<&str> for ElemType {
    fn from(value: &str) -> Self {
        match value {
            x if x == std::any::type_name::<f32>() => Self::F32,
            #[cfg(feature = "ffi_vec")]
            x if x == std::any::type_name::<Vector<f32, 1>>() => Self::F32Vec1,
            #[cfg(feature = "ffi_vec")]
            x if x == std::any::type_name::<Vector<f32, 2>>() => Self::F32Vec2,
            #[cfg(feature = "ffi_vec")]
            x if x == std::any::type_name::<Vector<f32, 3>>() => Self::F32Vec3,
            #[cfg(feature = "ffi_vec")]
            x if x == std::any::type_name::<Vector<f32, 4>>() => Self::F32Vec4,

            x if x == std::any::type_name::<f64>() => Self::F64,
            #[cfg(feature = "ffi_vec")]
            x if x == std::any::type_name::<Vector<f64, 1>>() => Self::F64Vec1,
            #[cfg(feature = "ffi_vec")]
            x if x == std::any::type_name::<Vector<f64, 2>>() => Self::F64Vec2,
            #[cfg(feature = "ffi_vec")]
            x if x == std::any::type_name::<Vector<f64, 3>>() => Self::F64Vec3,
            #[cfg(feature = "ffi_vec")]
            x if x == std::any::type_name::<Vector<f64, 4>>() => Self::F64Vec4,
            _ => unimplemented!(),
        }
    }
}

/// Info pertaining to a decoder.
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DecoderInfo {
    elem_type: ElemType,
    dims: OwnedCSlice<usize>,
}

/// Definition of a volume fetcher.
#[repr(C)]
#[allow(clippy::type_complexity, missing_debug_implementations)]
pub struct VolumeFetcher<'a, T> {
    ctx: VolumeFetcherCtx<T>,
    drop: unsafe extern "C" fn(VolumeFetcherCtx<T>),
    call: unsafe extern "C" fn(VolumeFetcherCtx<T>, CSlice<'_, usize>) -> T,
    _phantom: PhantomData<dyn Fn(CSlice<'_, usize>) -> T + Sync + Send + 'a>,
}

#[repr(C)]
union VolumeFetcherCtx<T> {
    opaque: *mut (),
    func: unsafe extern "C" fn(CSlice<'_, usize>) -> T,
}

impl<T> Copy for VolumeFetcherCtx<T> {}

impl<T> Clone for VolumeFetcherCtx<T> {
    fn clone(&self) -> Self {
        unsafe { std::ptr::read(self) }
    }
}

impl<'a, T> VolumeFetcher<'a, T> {
    fn call(&self, index: &[usize]) -> T {
        unsafe { (self.call)(self.ctx, index.into()) }
    }
}

impl<'a, T> Drop for VolumeFetcher<'a, T> {
    fn drop(&mut self) {
        unsafe { (self.drop)(self.ctx) }
    }
}

unsafe impl<'a, T: Send> Send for VolumeFetcher<'a, T> {}
unsafe impl<'a, T: Sync> Sync for VolumeFetcher<'a, T> {}

/// Definition of a block reader.
#[repr(C)]
#[allow(clippy::type_complexity, missing_debug_implementations)]
pub struct BlockReader<'a, T> {
    ctx: BlockReaderCtx<T>,
    drop: unsafe extern "C" fn(BlockReaderCtx<T>),
    call: unsafe extern "C" fn(BlockReaderCtx<T>, CSlice<'_, usize>) -> T,
    _phantom: PhantomData<dyn Fn(CSlice<'_, usize>) -> T + 'a>,
}

#[repr(C)]
union BlockReaderCtx<T> {
    opaque: *mut (),
    func: unsafe extern "C" fn(CSlice<'_, usize>) -> T,
}

impl<T> Copy for BlockReaderCtx<T> {}

impl<T> Clone for BlockReaderCtx<T> {
    fn clone(&self) -> Self {
        unsafe { std::ptr::read(self) }
    }
}

impl<'a, T> BlockReader<'a, T> {
    fn call(&self, index: &[usize]) -> T {
        unsafe { (self.call)(self.ctx, index.into()) }
    }
}

impl<'a, T> Drop for BlockReader<'a, T> {
    fn drop(&mut self) {
        unsafe { (self.drop)(self.ctx) }
    }
}

/// Definition of a block reader fetcher.
#[repr(C)]
#[allow(clippy::type_complexity, missing_debug_implementations)]
pub struct BlockReaderFetcher<'a, T> {
    ctx: BlockReaderFetcherCtx<'a, T>,
    drop: unsafe extern "C" fn(BlockReaderFetcherCtx<'a, T>),
    call: unsafe extern "C" fn(BlockReaderFetcherCtx<'a, T>, usize) -> BlockReader<'a, T>,
    _phantom: PhantomData<dyn Fn(usize) -> BlockReader<'a, T> + Sync + 'a>,
}

#[repr(C)]
union BlockReaderFetcherCtx<'a, T> {
    opaque: *mut (),
    func: unsafe extern "C" fn(usize) -> BlockReader<'a, T>,
}

impl<T> Copy for BlockReaderFetcherCtx<'_, T> {}

impl<T> Clone for BlockReaderFetcherCtx<'_, T> {
    fn clone(&self) -> Self {
        unsafe { std::ptr::read(self) }
    }
}

impl<'a, T> BlockReaderFetcher<'a, T> {
    fn call(&self, index: usize) -> BlockReader<'a, T> {
        unsafe { (self.call)(self.ctx, index) }
    }
}

impl<'a, T> Drop for BlockReaderFetcher<'a, T> {
    fn drop(&mut self) {
        unsafe { (self.drop)(self.ctx) }
    }
}

unsafe impl<'a, T: Sync> Sync for BlockReaderFetcherCtx<'a, T> {}

/// Definition of a block reader fetcher builder.
#[repr(C)]
#[allow(clippy::type_complexity, missing_debug_implementations)]
pub struct BlockReaderFetcherBuilder<'a, T> {
    ctx: BlockReaderFetcherBuilderCtx<'a, T>,
    drop: unsafe extern "C" fn(BlockReaderFetcherBuilderCtx<'a, T>),
    call: unsafe extern "C" fn(
        BlockReaderFetcherBuilderCtx<'a, T>,
        CSlice<'_, usize>,
        CSlice<'_, usize>,
    ) -> BlockReaderFetcher<'a, T>,
    _phantom: PhantomData<
        dyn FnOnce(CSlice<'_, usize>, CSlice<'_, usize>) -> BlockReaderFetcher<'a, T> + 'a,
    >,
}

#[repr(C)]
union BlockReaderFetcherBuilderCtx<'a, T> {
    opaque: *mut (),
    func: unsafe extern "C" fn(CSlice<'_, usize>, CSlice<'_, usize>) -> BlockReaderFetcher<'a, T>,
}

impl<T> Copy for BlockReaderFetcherBuilderCtx<'_, T> {}

impl<T> Clone for BlockReaderFetcherBuilderCtx<'_, T> {
    fn clone(&self) -> Self {
        unsafe { std::ptr::read(self) }
    }
}

impl<'a, T> BlockReaderFetcherBuilder<'a, T> {
    fn call(self, num_blocks: &[usize], block_size: &[usize]) -> BlockReaderFetcher<'a, T> {
        let this = ManuallyDrop::new(self);
        unsafe { (this.call)(this.ctx, num_blocks.into(), block_size.into()) }
    }
}

impl<'a, T> Drop for BlockReaderFetcherBuilder<'a, T> {
    fn drop(&mut self) {
        unsafe { (self.drop)(self.ctx) }
    }
}

/// Definition of a block writer.
#[repr(C)]
#[allow(clippy::type_complexity, missing_debug_implementations)]
pub struct BlockWriter<'a, T> {
    ctx: BlockWriterCtx<T>,
    drop: unsafe extern "C" fn(BlockWriterCtx<T>),
    call: unsafe extern "C" fn(BlockWriterCtx<T>, CSlice<'_, usize>, T),
    _phantom: PhantomData<dyn FnMut(CSlice<'_, usize>, T) + 'a>,
}

#[repr(C)]
union BlockWriterCtx<T> {
    opaque: *mut (),
    func: unsafe extern "C" fn(CSlice<'_, usize>, T),
}

impl<T> Copy for BlockWriterCtx<T> {}

impl<T> Clone for BlockWriterCtx<T> {
    fn clone(&self) -> Self {
        unsafe { std::ptr::read(self) }
    }
}

impl<'a, T> BlockWriter<'a, T> {
    fn call(&mut self, index: &[usize], val: T) {
        unsafe { (self.call)(self.ctx, index.into(), val) }
    }
}

impl<'a, T> Drop for BlockWriter<'a, T> {
    fn drop(&mut self) {
        unsafe { (self.drop)(self.ctx) }
    }
}

/// Definition of a block writer fetcher.
#[repr(C)]
#[allow(clippy::type_complexity, missing_debug_implementations)]
pub struct BlockWriterFetcher<'a, T> {
    ctx: BlockWriterFetcherCtx<'a, T>,
    drop: unsafe extern "C" fn(BlockWriterFetcherCtx<'a, T>),
    call: unsafe extern "C" fn(BlockWriterFetcherCtx<'a, T>, usize) -> BlockWriter<'a, T>,
    _phantom: PhantomData<dyn Fn(usize) -> BlockWriter<'a, T> + Sync + 'a>,
}

#[repr(C)]
union BlockWriterFetcherCtx<'a, T> {
    opaque: *mut (),
    func: unsafe extern "C" fn(usize) -> BlockWriter<'a, T>,
}

impl<T> Copy for BlockWriterFetcherCtx<'_, T> {}

impl<T> Clone for BlockWriterFetcherCtx<'_, T> {
    fn clone(&self) -> Self {
        unsafe { std::ptr::read(self) }
    }
}

impl<'a, T> BlockWriterFetcher<'a, T> {
    fn call(&self, index: usize) -> BlockWriter<'a, T> {
        unsafe { (self.call)(self.ctx, index) }
    }
}

impl<'a, T> Drop for BlockWriterFetcher<'a, T> {
    fn drop(&mut self) {
        unsafe { (self.drop)(self.ctx) }
    }
}

unsafe impl<'a, T: Sync> Sync for BlockWriterFetcher<'a, T> {}

/// Definition of a block writer fetcher builder.
#[repr(C)]
#[allow(clippy::type_complexity, missing_debug_implementations)]
pub struct BlockWriterFetcherBuilder<'a, T> {
    ctx: BlockWriterFetcherBuilderCtx<'a, T>,
    drop: unsafe extern "C" fn(BlockWriterFetcherBuilderCtx<'a, T>),
    call: unsafe extern "C" fn(
        BlockWriterFetcherBuilderCtx<'a, T>,
        CSlice<'_, usize>,
        CSlice<'_, usize>,
    ) -> BlockWriterFetcher<'a, T>,
    _phantom: PhantomData<
        dyn FnOnce(CSlice<'_, usize>, CSlice<'_, usize>) -> BlockWriterFetcher<'a, T> + 'a,
    >,
}

#[repr(C)]
union BlockWriterFetcherBuilderCtx<'a, T> {
    opaque: *mut (),
    func: unsafe extern "C" fn(CSlice<'_, usize>, CSlice<'_, usize>) -> BlockWriterFetcher<'a, T>,
}

impl<T> Copy for BlockWriterFetcherBuilderCtx<'_, T> {}

impl<T> Clone for BlockWriterFetcherBuilderCtx<'_, T> {
    fn clone(&self) -> Self {
        unsafe { std::ptr::read(self) }
    }
}

impl<'a, T> BlockWriterFetcherBuilder<'a, T> {
    fn call(self, num_blocks: &[usize], block_size: &[usize]) -> BlockWriterFetcher<'a, T> {
        let this = ManuallyDrop::new(self);
        unsafe { (this.call)(this.ctx, num_blocks.into(), block_size.into()) }
    }
}

impl<'a, T> Drop for BlockWriterFetcherBuilder<'a, T> {
    fn drop(&mut self) {
        unsafe { (self.drop)(self.ctx) }
    }
}

/////////////////////////////// Encoder ///////////////////////////////

macro_rules! encoder_def {
    ($($T:ty);*) => {
        $(
            encoder_def! { $T, wavelet_rs_encoder_ $T }

            #[cfg(feature = "ffi_vec")]
            encoder_def! { Vector<$T, 1>, wavelet_rs_encoder_vec_1_ $T }
            #[cfg(feature = "ffi_vec")]
            encoder_def! { Vector<$T, 2>, wavelet_rs_encoder_vec_2_ $T }
            #[cfg(feature = "ffi_vec")]
            encoder_def! { Vector<$T, 3>, wavelet_rs_encoder_vec_3_ $T }
            #[cfg(feature = "ffi_vec")]
            encoder_def! { Vector<$T, 4>, wavelet_rs_encoder_vec_4_ $T }
        )*
    };

    ($T:ty, $($N:tt)*) => {
        paste! {
            /// Constructs a new encoder.
            #[no_mangle]
            pub unsafe extern "C" fn [<$($N)* _new>](
                dims: *const CSlice<'_, usize>,
                num_base_dims: usize,
            ) -> Box<VolumeWaveletEncoder<'static, $T>> {
                Box::new(VolumeWaveletEncoder::new(&*dims, num_base_dims))
            }

            /// Deallocates and destructs an encoder.
            #[no_mangle]
            pub extern "C" fn [<$($N)* _free>](
                _: Box<VolumeWaveletEncoder<'_, $T>>,
            ) {}

            /// Adds a volume fetcher to the encoder.
            #[no_mangle]
            pub unsafe extern "C" fn [<$($N)* _add_fetcher>]<'a, 'b: 'a>(
                encoder: *mut VolumeWaveletEncoder<'a, $T>,
                index: *const CSlice<'_, usize>,
                fetcher: *const MaybeUninit<VolumeFetcher<'b, $T>>,
            ) {
                let fetcher = (*fetcher).assume_init_read();
                let f = move |index: &[usize]| fetcher.call(index);
                (*encoder).add_fetcher(&*index, f);
            }

            /// Encodes the dataset with the specified block size and the haar wavelet.
            #[no_mangle]
            pub unsafe extern "C" fn [<$($N)* _encode_haar>](
                encoder: *const VolumeWaveletEncoder<'_, $T>,
                output: *const std::os::raw::c_char,
                block_size: *const CSlice<'_, usize>,
            ) {
                let output = CStr::from_ptr(output.cast());
                let output = String::from_utf8_lossy(output.as_ref().to_bytes()).into_owned();
                (*encoder).encode(output, &*block_size, HaarWavelet)
            }

            /// Encodes the dataset with the specified block size and the average filter.
            #[no_mangle]
            pub unsafe extern "C" fn [<$($N)* _encode_average>](
                encoder: *const VolumeWaveletEncoder<'_, $T>,
                output: *const std::os::raw::c_char,
                block_size: *const CSlice<'_, usize>,
            ) {
                let output = CStr::from_ptr(output.cast());
                let output = String::from_utf8_lossy(output.as_ref().to_bytes()).into_owned();
                (*encoder).encode(output, &*block_size, AverageFilter)
            }
        }

        encoder_def! { metadata $T, u8, $($N)* }
        encoder_def! { metadata $T, u16, $($N)* }
        encoder_def! { metadata $T, u32, $($N)* }
        encoder_def! { metadata $T, u64, $($N)* }
        encoder_def! { metadata $T, i8, $($N)* }
        encoder_def! { metadata $T, i16, $($N)* }
        encoder_def! { metadata $T, i32, $($N)* }
        encoder_def! { metadata $T, i64, $($N)* }
        encoder_def! { metadata $T, f32, $($N)* }
        encoder_def! { metadata $T, f64, $($N)* }
        encoder_def! { metadata $T, CString, $($N)* }
    };

    (metadata $T:ty, $U:ty, $($N:tt)*) => {
        encoder_def! { metadata_ $T, $U, ($($N)*); $U }

        #[cfg(feature = "ffi_metadata_arr")]
        encoder_def! { metadata_ $T, CArray<$U, 1>, ($($N)*); $U _arr_1 }
        #[cfg(feature = "ffi_metadata_arr")]
        encoder_def! { metadata_ $T, CArray<$U, 2>, ($($N)*); $U _arr_2 }
        #[cfg(feature = "ffi_metadata_arr")]
        encoder_def! { metadata_ $T, CArray<$U, 3>, ($($N)*); $U _arr_3 }
        #[cfg(feature = "ffi_metadata_arr")]
        encoder_def! { metadata_ $T, CArray<$U, 4>, ($($N)*); $U _arr_4 }

        #[cfg(feature = "ffi_metadata_slice")]
        encoder_def! { metadata_ $T, OwnedCSlice<$U>, ($($N)*); $U _slice }
    };

    (metadata_ $T:ty, $U:ty, ( $($N:tt)*); $($M:tt)*) => {
        paste! {
            /// Fetches a value inserted into the metadata.
            #[no_mangle]
            pub unsafe extern "C" fn [<$($N)* _metadata_get_ $($M)*>](
                encoder: *const VolumeWaveletEncoder<'_, $T>,
                key: *const std::ffi::c_char,
                out: *mut MaybeUninit<COption<$U>>
            ) {
                let key = CStr::from_ptr(key.cast());
                let key = String::from_utf8_lossy(key.as_ref().to_bytes());
                let res = (*encoder).get_metadata(&*key).into();
                (*out).write(res);
            }

            /// Inserts some metadata which will be included into the encoded dataset.
            #[no_mangle]
            pub unsafe extern "C" fn [<$($N)* _metadata_insert_ $($M)*>](
                encoder: *mut VolumeWaveletEncoder<'_, $T>,
                key: *const std::ffi::c_char,
                value: *const MaybeUninit<$U>
            ) -> u8 {
                let key = CStr::from_ptr(key.cast());
                let key = String::from_utf8_lossy(key.as_ref().to_bytes()).into_owned();
                let value = (*value).assume_init_read();
                (*encoder).insert_metadata(key, value) as u8
            }
        }
    };
}

encoder_def! {
    f32; f64
}

/////////////////////////////// Decoder ///////////////////////////////

macro_rules! decoder_def {
    ($($T:ty);*) => {
        $(
            decoder_def! { impl_ $T, wavelet_rs_decoder_ $T }

            #[cfg(feature = "ffi_vec")]
            decoder_def! { impl_ Vector<$T, 1>, wavelet_rs_decoder_vec_1_ $T }
            #[cfg(feature = "ffi_vec")]
            decoder_def! { impl_ Vector<$T, 2>, wavelet_rs_decoder_vec_2_ $T }
            #[cfg(feature = "ffi_vec")]
            decoder_def! { impl_ Vector<$T, 3>, wavelet_rs_decoder_vec_3_ $T }
            #[cfg(feature = "ffi_vec")]
            decoder_def! { impl_ Vector<$T, 4>, wavelet_rs_decoder_vec_4_ $T }
        )*
    };

    (impl_ $T:ty, $($N:tt)*) => {
        paste! {
            /// Constructs a new decoder.
            #[no_mangle]
            pub unsafe extern "C" fn [<$($N)* _new>](
                input: *const std::os::raw::c_char,
            ) -> Box<VolumeWaveletDecoder<$T>> {
                let input = CStr::from_ptr(input.cast());
                let input = String::from_utf8_lossy(input.as_ref().to_bytes());
                Box::new(VolumeWaveletDecoder::new(&*input))
            }

            /// Deallocates and destructs an decoder.
            #[no_mangle]
            pub extern "C" fn [<$($N)* _free>](
                _: Box<VolumeWaveletDecoder<$T>>,
            ) {}

            /// Fetches the dimensions of the encoded dataset.
            #[no_mangle]
            pub unsafe extern "C" fn [<$($N)* _dims>](
                decoder: *const VolumeWaveletDecoder<$T>,
                output: *mut MaybeUninit<CSlice<'_, usize>>
            ) {
                let dims = (*decoder).dims().into();
                (*output).write(dims);
            }

            /// Fetches the blocksize used to encode the dataset.
            #[no_mangle]
            pub unsafe extern "C" fn [<$($N)* _block_size>](
                decoder: *const VolumeWaveletDecoder<$T>,
                output: *mut MaybeUninit<CSlice<'_, usize>>
            ) {
                let size = (*decoder).block_size().into();
                (*output).write(size);
            }

            /// Fetches the number of blocks used for encoding the dataset.
            #[no_mangle]
            pub unsafe extern "C" fn [<$($N)* _block_counts>](
                decoder: *const VolumeWaveletDecoder<$T>,
                output: *mut MaybeUninit<CSlice<'_, usize>>
            ) {
                let counts = (*decoder).block_counts().into();
                (*output).write(counts);
            }

            /// Decodes the dataset.
            #[no_mangle]
            pub unsafe extern "C" fn [<$($N)* _decode>](
                decoder: *const VolumeWaveletDecoder<$T>,
                writer_fetcher: *const MaybeUninit<BlockWriterFetcherBuilder<'_, $T>>,
                roi: *const CSlice<'_, CRange<usize>>,
                levels: *const CSlice<'_, u32>
            ) {
                let writer_fetcher = (*writer_fetcher).assume_init_read();
                let writer_fetcher = move |index: &[usize], block_size: &[usize]| {
                    let fetcher = writer_fetcher.call(index, block_size);
                    move |index: usize| {
                        let mut writer = fetcher.call(index);
                        move |index: &[usize], val: $T| writer.call(index, val)
                    }
                };

                let roi: Vec<Range<_>> = (*roi).iter().cloned().map(|r| r.into()).collect();
                (*decoder).decode(writer_fetcher, &roi, &*levels);
            }

            /// Applies a partial decoding to a partially decoded dataset.
            #[no_mangle]
            pub unsafe extern "C" fn [<$($N)* _refine>](
                decoder: *const VolumeWaveletDecoder<$T>,
                reader_fetcher: *const MaybeUninit<BlockReaderFetcherBuilder<'_, $T>>,
                writer_fetcher: *const MaybeUninit<BlockWriterFetcherBuilder<'_, $T>>,
                input_range: *const CSlice<'_, CRange<usize>>,
                output_range: *const CSlice<'_, CRange<usize>>,
                curr_levels: *const CSlice<'_, u32>,
                refinements: *const CSlice<'_, u32>
            ) {
                let reader_fetcher = (*reader_fetcher).assume_init_read();
                let writer_fetcher = (*writer_fetcher).assume_init_read();
                let reader_fetcher = move |index: &[usize], block_size: &[usize]| {
                    let fetcher = reader_fetcher.call(index, block_size);
                    move |index: usize| {
                        let reader = fetcher.call(index);
                        move |index: &[usize]| reader.call(index)
                    }
                };

                let writer_fetcher = move |index: &[usize], block_size: &[usize]| {
                    let fetcher = writer_fetcher.call(index, block_size);
                    move |index: usize| {
                        let mut writer = fetcher.call(index);
                        move |index: &[usize], val: $T| writer.call(index, val)
                    }
                };

                let input_range: Vec<Range<_>> = (*input_range)
                    .iter()
                    .cloned()
                    .map(|r| r.into())
                    .collect();

                let output_range: Vec<Range<_>> = (*output_range)
                    .iter()
                    .cloned()
                    .map(|r| r.into())
                    .collect();
                (*decoder).refine(
                    reader_fetcher,
                    writer_fetcher,
                    &input_range,
                    &output_range,
                    &*curr_levels,
                    &*refinements
                );
            }
        }

        decoder_def! { metadata $T, u8, $($N)* }
        decoder_def! { metadata $T, u16, $($N)* }
        decoder_def! { metadata $T, u32, $($N)* }
        decoder_def! { metadata $T, u64, $($N)* }
        decoder_def! { metadata $T, i8, $($N)* }
        decoder_def! { metadata $T, i16, $($N)* }
        decoder_def! { metadata $T, i32, $($N)* }
        decoder_def! { metadata $T, i64, $($N)* }
        decoder_def! { metadata $T, f32, $($N)* }
        decoder_def! { metadata $T, f64, $($N)* }
        decoder_def! { metadata $T, CString, $($N)* }
    };

    (metadata $T:ty, $U:ty, $($N:tt)*) => {
        decoder_def! { metadata_ $T, $U, ($($N)*); $U }

        #[cfg(feature = "ffi_metadata_arr")]
        decoder_def! { metadata_ $T, CArray<$U, 1>, ($($N)*); $U _arr_1 }
        #[cfg(feature = "ffi_metadata_arr")]
        decoder_def! { metadata_ $T, CArray<$U, 2>, ($($N)*); $U _arr_2 }
        #[cfg(feature = "ffi_metadata_arr")]
        decoder_def! { metadata_ $T, CArray<$U, 3>, ($($N)*); $U _arr_3 }
        #[cfg(feature = "ffi_metadata_arr")]
        decoder_def! { metadata_ $T, CArray<$U, 4>, ($($N)*); $U _arr_4 }

        #[cfg(feature = "ffi_metadata_slice")]
        decoder_def! { metadata_ $T, OwnedCSlice<$U>, ($($N)*); $U _slice }
    };

    (metadata_ $T:ty, $U:ty, ( $($N:tt)*); $($M:tt)*) => {
        paste! {
            /// Fetches a value inserted into the metadata.
            #[no_mangle]
            pub unsafe extern "C" fn [<$($N)* _metadata_get_ $($M)*>](
                decoder: *const VolumeWaveletDecoder<$T>,
                key: *const std::ffi::c_char,
                res: *mut MaybeUninit<COption<$U>>
            ) {
                let key = CStr::from_ptr(key.cast());
                let key = String::from_utf8_lossy(key.as_ref().to_bytes());
                let x = (*decoder).get_metadata(&*key).into();
                (*res).write(x);
            }
        }
    };
}

decoder_def! {
    f32; f64
}

/// Returns some info pertaining to the encoded data located at `path`.
#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn wavelet_rs_get_decoder_info(
    path: *const std::ffi::c_char,
    output: *mut MaybeUninit<DecoderInfo>,
) {
    use crate::encoder::OutputHeader;
    use crate::stream::DeserializeStream;

    let path = CStr::from_ptr(path.cast());
    let path = String::from_utf8_lossy(path.as_ref().to_bytes());

    let f = std::fs::File::open(&*path).unwrap();
    let stream = DeserializeStream::new_decode(f).unwrap();
    let mut stream = stream.stream();
    let (elem_type, dims) = OutputHeader::<()>::deserialize_info(&mut stream);

    let elem_type: ElemType = (*elem_type).into();
    let dims: OwnedCSlice<usize> = dims.into_boxed_slice().into();

    let info = DecoderInfo { elem_type, dims };

    (*output).write(info);
}

/// Returns a list of block sizes which satisfies the given error constrains.
///
/// A `L0` error occurs when the size of the second pass is not a power of two,
/// while a `L1` error occurs when the block size does not divide into `dim`.
#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn wavelet_rs_allowed_block_sizes(
    dim: usize,
    allow_l0_error: u8,
    allow_l1_error: u8,
    output: *mut MaybeUninit<COption<OwnedCSlice<usize>>>,
) {
    let sizes = crate::encoder::allowed_block_sizes(dim, allow_l0_error != 0, allow_l1_error != 0);
    let sizes: Option<OwnedCSlice<usize>> = sizes.map(|sizes| sizes.into());

    (*output).write(sizes.into());
}

/// Returns the minimum level required to avoid any error.
///
/// A `L0` error occurs when the size of the second pass is not a power of two,
/// while a `L1` error occurs when the block size does not divide into `dim`.
#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn wavelet_rs_min_level(
    dim: usize,
    block_size: usize,
    allow_l0_error: u8,
    allow_l1_error: u8,
) -> u32 {
    crate::decoder::min_level(dim, block_size, allow_l0_error != 0, allow_l1_error != 0)
}

/// Returns the maximum range which does not produce any unwanted error.
///
/// A `L1` error occurs when the block size does not divide into `dim`.
#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn wavelet_rs_max_range(
    dim: usize,
    block_size: usize,
    allow_l1_error: u8,
    output: *mut MaybeUninit<CRange<usize>>,
) {
    let range = crate::decoder::max_range(dim, block_size, allow_l1_error != 0);
    (*output).write(range.into());
}
