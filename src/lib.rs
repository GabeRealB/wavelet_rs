//! Implementation of wavelet based data processing utilities.
#![warn(
    missing_docs,
    rust_2018_idioms,
    missing_debug_implementations,
    rustdoc::broken_intra_doc_links
)]
#![feature(slice_ptr_get)]
#![feature(int_log)]

pub mod decoder;
pub mod encoder;
pub mod ffi;
pub mod filter;
pub mod range;
pub mod stream;
pub mod transformations;
pub mod vector;
pub mod volume;

pub(crate) mod utilities;
