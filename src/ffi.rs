//! C-API of the library.

use std::{
    ffi::{c_char, CStr},
    fmt::Debug,
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use paste::paste;

use crate::{
    encoder::VolumeWaveletEncoder,
    filter::{AverageFilter, HaarWavelet},
    stream::{Deserializable, Serializable},
};

#[cfg(any(feature = "ffi_vec", feature = "ffi_mat"))]
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
    u8; u16; u32; u64; i8; i16; i32; i64; f32; f64; c_char
}

/// C compatible owned string.
#[repr(C)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
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

unsafe impl<'a, T> Send for VolumeFetcher<'a, T> {}
unsafe impl<'a, T> Sync for VolumeFetcher<'a, T> {}

/////////////////////////////// Encoder ///////////////////////////////

macro_rules! encoder_def {
    ($($T:ty);*) => {
        $(
            encoder_def! { $T, wavelet_rs_encoder_ $T }
            /* encoder_vec_def! { $T } */

            #[cfg(feature = "ffi_vec")]
            encoder_def! { Vector<$T, 1>, wavelet_rs_encoder_vec_1_ $T }
            #[cfg(feature = "ffi_vec")]
            encoder_def! { Vector<$T, 2>, wavelet_rs_encoder_vec_2_ $T }
            #[cfg(feature = "ffi_vec")]
            encoder_def! { Vector<$T, 3>, wavelet_rs_encoder_vec_3_ $T }
            #[cfg(feature = "ffi_vec")]
            encoder_def! { Vector<$T, 4>, wavelet_rs_encoder_vec_4_ $T }

            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 1>, 1>, wavelet_rs_encoder_mat_1x1_ $T }
            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 2>, 1>, wavelet_rs_encoder_mat_1x2_ $T }
            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 3>, 1>, wavelet_rs_encoder_mat_1x3_ $T }
            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 4>, 1>, wavelet_rs_encoder_mat_1x4_ $T }

            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 1>, 2>, wavelet_rs_encoder_mat_2x1_ $T }
            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 2>, 2>, wavelet_rs_encoder_mat_2x2_ $T }
            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 3>, 2>, wavelet_rs_encoder_mat_2x3_ $T }
            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 4>, 2>, wavelet_rs_encoder_mat_2x4_ $T }

            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 1>, 3>, wavelet_rs_encoder_mat_3x1_ $T }
            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 2>, 3>, wavelet_rs_encoder_mat_3x2_ $T }
            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 3>, 3>, wavelet_rs_encoder_mat_3x3_ $T }
            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 4>, 3>, wavelet_rs_encoder_mat_3x4_ $T }

            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 1>, 4>, wavelet_rs_encoder_mat_4x1_ $T }
            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 2>, 4>, wavelet_rs_encoder_mat_4x2_ $T }
            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 3>, 4>, wavelet_rs_encoder_mat_4x3_ $T }
            #[cfg(feature = "ffi_mat")]
            encoder_def! { Vector<Vector<$T, 4>, 4>, wavelet_rs_encoder_mat_4x4_ $T }
        )*
    };

    ($T:ty, $($N:tt)*) => {
        paste! {
            /// Constructs a new encoder.
            #[no_mangle]
            pub extern "C" fn [<$($N)* _new>](
                dims: CSlice<'_, usize>,
                num_base_dims: usize,
            ) -> Box<VolumeWaveletEncoder<'static, $T>> {
                Box::new(VolumeWaveletEncoder::new(&dims, num_base_dims))
            }

            /// Deallocates and destructs an encoder.
            #[no_mangle]
            pub extern "C" fn [<$($N)* _free>](
                _: Box<VolumeWaveletEncoder<'_, $T>>,
            ) {}

            /// Adds a volume fetcher to the encoder.
            #[no_mangle]
            pub extern "C" fn [<$($N)* _add_fetcher>]<'a, 'b: 'a>(
                mut encoder: NonNull<VolumeWaveletEncoder<'a, $T>>,
                index: CSlice<'_, usize>,
                fetcher: VolumeFetcher<'b, $T>,
            ) {
                let f = move |index: &[usize]| fetcher.call(index);

                unsafe {
                    encoder.as_mut().add_fetcher(&index, f);
                }
            }

            /// Encodes the dataset with the specified block size and the haar wavelet.
            #[no_mangle]
            pub extern "C" fn [<$($N)* _encode_haar>](
                encoder: NonNull<VolumeWaveletEncoder<'_, $T>>,
                output: *const std::os::raw::c_char,
                block_size: CSlice<'_, usize>,
            ) {
                unsafe {
                    let output = CStr::from_ptr(output.cast());
                    let output = String::from_utf8_lossy(output.as_ref().to_bytes()).into_owned();
                    encoder.as_ref().encode(output, &block_size, HaarWavelet)
                }
            }

            /// Encodes the dataset with the specified block size and the average filter.
            #[no_mangle]
            pub extern "C" fn [<$($N)* _encode_average>](
                encoder: NonNull<VolumeWaveletEncoder<'_, $T>>,
                output: *const std::os::raw::c_char,
                block_size: CSlice<'_, usize>,
            ) {
                unsafe {
                    let output = CStr::from_ptr(output.cast());
                    let output = String::from_utf8_lossy(output.as_ref().to_bytes()).into_owned();
                    encoder.as_ref().encode(output, &block_size, AverageFilter)
                }
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

        encoder_def! { metadata_ $T, CString, ($($N)*); string}
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
            pub extern "C" fn [<$($N)* _metadata_get_ $($M)*>](
                encoder: NonNull<VolumeWaveletEncoder<'_, $T>>,
                key: *const std::ffi::c_char,
            ) -> COption<$U> {
                unsafe {
                    let key = CStr::from_ptr(key.cast());
                    let key = String::from_utf8_lossy(key.as_ref().to_bytes());
                    encoder.as_ref().get_metadata(&*key).into()
                }
            }

            /// Inserts some metadata which will be included into the encoded dataset.
            #[no_mangle]
            pub extern "C" fn [<$($N)* _metadata_insert_ $($M)*>](
                mut encoder: NonNull<VolumeWaveletEncoder<'_, $T>>,
                key: *const std::ffi::c_char,
                value: $U
            ) -> u8 {
                unsafe {
                    let key = CStr::from_ptr(key.cast());
                    let key = String::from_utf8_lossy(key.as_ref().to_bytes()).into_owned();
                    encoder.as_mut().insert_metadata(key, value) as u8
                }
            }
        }
    };
}

encoder_def! {
    f32; f64
}
