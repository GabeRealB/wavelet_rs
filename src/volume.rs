//! Volume utilities.

use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use alloca::with_alloca_zeroed;
use num_traits::{Float, Zero};
use thiserror::Error;

use crate::utilities::{
    flatten_idx, flatten_idx_with_offset, flatten_idx_with_offset_unchecked, strides_for_dims,
};

/// Multi-dimensional volume.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct VolumeBlock<T> {
    data: Vec<T>,
    dims: Vec<usize>,
    strides: Vec<usize>,
}

impl<T: Clone> VolumeBlock<T> {
    /// Constructs a new block filled with the provided value.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let block = VolumeBlock::<f32>::new_fill(&dims, 1.0).unwrap();
    ///
    /// assert_eq!(block[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(block[[1usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(block[[0usize, 1usize].as_ref()], 1.0);
    /// assert_eq!(block[[1usize, 1usize].as_ref()], 1.0);
    /// ```
    pub fn new_fill(dims: &[usize], fill: T) -> Result<Self, VolumeError> {
        let num_elements = dims.iter().product();
        let data = vec![fill; num_elements];
        Self::new_with_data(dims, data)
    }
}

impl<T: Zero + Clone> VolumeBlock<T> {
    /// Constructs a new zeroed block.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let block = VolumeBlock::<f32>::new_zero(&dims).unwrap();
    ///
    /// assert_eq!(block[[0usize, 0usize].as_ref()], 0.0);
    /// assert_eq!(block[[1usize, 0usize].as_ref()], 0.0);
    /// assert_eq!(block[[0usize, 1usize].as_ref()], 0.0);
    /// assert_eq!(block[[1usize, 1usize].as_ref()], 0.0);
    /// ```
    pub fn new_zero(dims: &[usize]) -> Result<Self, VolumeError> {
        Self::new_fill(dims, T::zero())
    }
}

impl<T: Default + Clone> VolumeBlock<T> {
    /// Constructs a new zeroed block.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let block = VolumeBlock::<f32>::new_default(&dims).unwrap();
    ///
    /// assert_eq!(block[[0usize, 0usize].as_ref()], 0.0);
    /// assert_eq!(block[[1usize, 0usize].as_ref()], 0.0);
    /// assert_eq!(block[[0usize, 1usize].as_ref()], 0.0);
    /// assert_eq!(block[[1usize, 1usize].as_ref()], 0.0);
    /// ```
    pub fn new_default(dims: &[usize]) -> Result<Self, VolumeError> {
        Self::new_fill(dims, Default::default())
    }
}

impl<T> VolumeBlock<T> {
    /// Constructs a new block.
    ///
    /// # Examples
    ////newegg
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let block = VolumeBlock::new_with_data(&dims, data).unwrap();
    ///
    /// assert_eq!(block[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(block[[1usize, 0usize].as_ref()], 2.0);
    /// assert_eq!(block[[0usize, 1usize].as_ref()], 3.0);
    /// assert_eq!(block[[1usize, 1usize].as_ref()], 4.0);
    /// ```
    pub fn new_with_data(dims: &[usize], data: Vec<T>) -> Result<Self, VolumeError> {
        let num_elements = dims.iter().product();

        if dims.is_empty() {
            Err(VolumeError::ZeroBlockLength)
        } else if let Some(len) = dims.iter().find(|&&len| len == 0) {
            Err(VolumeError::InvalidDimensionLenght { length: *len })
        } else if num_elements != data.len() {
            Err(VolumeError::InvalidNumberOfElements {
                got: data.len(),
                required: num_elements,
            })
        } else {
            let strides = strides_for_dims(dims);

            Ok(Self {
                data,
                dims: dims.into(),
                strides,
            })
        }
    }

    /// Returns the dimensions of the block.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Constructs a new window into the block.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let block = VolumeBlock::new_with_data(&dims, data).unwrap();
    /// let window = block.window();
    ///
    /// assert_eq!(window[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(window[[1usize, 0usize].as_ref()], 2.0);
    /// assert_eq!(window[[0usize, 1usize].as_ref()], 3.0);
    /// assert_eq!(window[[1usize, 1usize].as_ref()], 4.0);
    /// ```
    pub fn window(&self) -> VolumeWindow<'_, T> {
        VolumeWindow {
            dims: self.dims.clone(),
            dim_offsets: vec![0; self.dims.len()],
            block_data: &*self.data,
            block_dims: &self.dims,
            block_strides: &self.strides,
            _phantom: PhantomData,
        }
    }

    /// Constructs a new mutable window into the block.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut block = VolumeBlock::new_with_data(&dims, data).unwrap();
    /// let window = block.window_mut();
    ///
    /// assert_eq!(window[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(window[[1usize, 0usize].as_ref()], 2.0);
    /// assert_eq!(window[[0usize, 1usize].as_ref()], 3.0);
    /// assert_eq!(window[[1usize, 1usize].as_ref()], 4.0);
    /// ```
    pub fn window_mut(&mut self) -> VolumeWindowMut<'_, T> {
        VolumeWindowMut {
            dims: self.dims.clone(),
            dim_offsets: vec![0; self.dims.len()],
            block_data: &mut *self.data,
            block_dims: &self.dims,
            block_strides: &self.strides,
            _phantom: PhantomData,
        }
    }

    /// Checks that the data contained inside the volumes is equal
    /// apart from a specified error value.
    pub fn is_equal(&self, other: &Self, eps: T) -> bool
    where
        T: Float,
    {
        self.dims == other.dims
            && self
                .data
                .iter()
                .zip(&other.data)
                .all(|(this, other)| (*this - *other).abs() < eps)
    }

    /// Returns a slice to the flat representation of the volume.
    pub fn flatten(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable slice to the flat representation of the volume.
    pub fn flatten_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns a reference to the element at the provided index.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Returns a mutable reference to the element at the provided index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)
    }

    /// Returns a reference to the element at the provided index.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined
    /// behavior even if the resulting reference is not used.
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        self.data.get_unchecked(index)
    }

    /// Returns a mutable reference to the element at the provided index.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined
    /// behavior even if the resulting reference is not used.
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        self.data.get_unchecked_mut(index)
    }
}

impl<T> Index<usize> for VolumeBlock<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for VolumeBlock<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T> Index<&[usize]> for VolumeBlock<T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        let idx = flatten_idx(&self.dims, &self.strides, index);
        &self[idx]
    }
}

impl<T> IndexMut<&[usize]> for VolumeBlock<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        let idx = flatten_idx(&self.dims, &self.strides, index);
        &mut self[idx]
    }
}

/// Errors that can occur when constructing a [`VolumeBlock`].
#[derive(Error, Debug)]
pub enum VolumeError {
    /// Tried to construct a [`VolumeBlock`] containing no elements.
    #[error("a block length of 0 is not supported")]
    ZeroBlockLength,

    /// Tried to construct a [`VolumeBlock`] with an invalid dimension length.
    #[error("invalid length for dimension (got {length})")]
    InvalidDimensionLenght {
        /// Length of the axis.
        length: usize,
    },

    /// Number of proveded elements does not match with the required length.
    #[error("invalid number of elements (got {got}, required {required})")]
    InvalidNumberOfElements {
        /// Number of provided elements.
        got: usize,
        /// Number of required elements.
        required: usize,
    },
}

/// A borrowed window into a [`VolumeBlock`].
#[derive(Debug, Clone)]
pub struct VolumeWindow<'a, T> {
    dims: Vec<usize>,
    dim_offsets: Vec<usize>,
    block_data: *const [T],
    block_dims: &'a [usize],
    block_strides: &'a [usize],
    _phantom: PhantomData<&'a [T]>,
}

impl<'a, T> VolumeWindow<'a, T> {
    /// Splits a window into two non overlapping halves.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let block = VolumeBlock::new_with_data(&dims, data).unwrap();
    /// let window = block.window();
    /// let (left, right) = window.split(0);
    ///
    /// assert_eq!(left[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(left[[0usize, 1usize].as_ref()], 3.0);
    ///
    /// assert_eq!(right[[0usize, 0usize].as_ref()], 2.0);
    /// assert_eq!(right[[0usize, 1usize].as_ref()], 4.0);
    ///
    /// let (top, bottom) = window.split(1);
    ///
    /// assert_eq!(top[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(top[[1usize, 0usize].as_ref()], 2.0);
    ///
    /// assert_eq!(bottom[[0usize, 0usize].as_ref()], 3.0);
    /// assert_eq!(bottom[[1usize, 0usize].as_ref()], 4.0);
    /// ```
    pub fn split(&self, dim: usize) -> (VolumeWindow<'_, T>, VolumeWindow<'_, T>) {
        assert!(self.dims[dim] % 2 == 0);
        let mut dims = self.dims.clone();
        let offsets_left = self.dim_offsets.clone();
        let mut offsets_right = offsets_left.clone();

        dims[dim] /= 2;
        offsets_right[dim] += dims[dim];

        let left = Self {
            dims: dims.clone(),
            dim_offsets: offsets_left,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        };

        let right = Self {
            dims,
            dim_offsets: offsets_right,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        };

        (left, right)
    }

    /// Splits a window into two non overlapping halves.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let block = VolumeBlock::new_with_data(&dims, data).unwrap();
    /// let window = block.window();
    /// let (left, right) = window.split_into(0);
    ///
    /// assert_eq!(left[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(left[[0usize, 1usize].as_ref()], 3.0);
    ///
    /// assert_eq!(right[[0usize, 0usize].as_ref()], 2.0);
    /// assert_eq!(right[[0usize, 1usize].as_ref()], 4.0);
    ///
    /// let window = block.window();
    /// let (top, bottom) = window.split_into(1);
    ///
    /// assert_eq!(top[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(top[[1usize, 0usize].as_ref()], 2.0);
    ///
    /// assert_eq!(bottom[[0usize, 0usize].as_ref()], 3.0);
    /// assert_eq!(bottom[[1usize, 0usize].as_ref()], 4.0);
    /// ```
    pub fn split_into(self, dim: usize) -> (VolumeWindow<'a, T>, VolumeWindow<'a, T>) {
        assert!(self.dims[dim] % 2 == 0);
        let mut dims = self.dims;
        let offsets_left = self.dim_offsets;
        let mut offsets_right = offsets_left.clone();

        dims[dim] /= 2;
        offsets_right[dim] += dims[dim];

        let left = Self {
            dims: dims.clone(),
            dim_offsets: offsets_left,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        };

        let right = Self {
            dims,
            dim_offsets: offsets_right,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        };

        (left, right)
    }

    /// Borrows a subwindow in the range `[0..dims[0], ...]` from the window.
    pub fn custom_range(&self, range: &[usize]) -> VolumeWindow<'_, T> {
        assert_eq!(self.dims.len(), range.len());
        assert!(range
            .iter()
            .zip(&self.dim_offsets)
            .zip(&self.dims)
            .all(|((&r, &o), &d)| r + o <= d));

        let dim_offsets = self.dim_offsets.clone();
        Self {
            dims: range.into(),
            dim_offsets,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        }
    }

    /// Borrows a subwindow in the range `[offset[0]..dims[0], ...]` from the window.
    pub fn custom_window(&self, offset: &[usize], dims: &[usize]) -> VolumeWindow<'_, T> {
        assert_eq!(self.dims.len(), offset.len());
        assert_eq!(self.dims.len(), dims.len());
        assert!(offset
            .iter()
            .zip(dims)
            .zip(&self.dim_offsets)
            .zip(&self.dims)
            .all(|(((&n_o, &n_d), &o), &d)| n_o + n_d + o <= d));

        let dim_offsets = offset
            .iter()
            .zip(&self.dim_offsets)
            .map(|(&n_o, &o)| n_o + o)
            .collect();
        Self {
            dims: dims.into(),
            dim_offsets,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        }
    }

    /// Returns the dimensions of the window.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Constructs an iterator over the rows of a window.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2, 2];
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let block = VolumeBlock::new_with_data(&dims, data).unwrap();
    /// let window = block.window();
    ///
    /// let mut rows_x = window.rows(0);
    /// assert_eq!(rows_x.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [1.0, 2.0]);
    /// assert_eq!(rows_x.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [3.0, 4.0]);
    /// assert_eq!(rows_x.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [5.0, 6.0]);
    /// assert_eq!(rows_x.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [7.0, 8.0]);
    /// assert!(rows_x.next().is_none());
    ///
    /// let mut rows_y = window.rows(1);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [1.0, 3.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [2.0, 4.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [5.0, 7.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [6.0, 8.0]);
    /// assert!(rows_y.next().is_none());
    ///
    /// let mut rows_z = window.rows(2);
    /// assert_eq!(rows_z.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [1.0, 5.0]);
    /// assert_eq!(rows_z.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [2.0, 6.0]);
    /// assert_eq!(rows_z.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [3.0, 7.0]);
    /// assert_eq!(rows_z.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [4.0, 8.0]);
    /// assert!(rows_z.next().is_none());
    ///
    /// let (top, _) = window.split(1);
    /// let mut rows_x = top.rows(0);
    /// assert_eq!(rows_x.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [1.0, 2.0]);
    /// assert_eq!(rows_x.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [5.0, 6.0]);
    /// assert!(rows_x.next().is_none());
    ///
    /// let mut rows_y = top.rows(1);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [1.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [2.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [5.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [6.0]);
    /// assert!(rows_y.next().is_none());
    ///
    /// let mut rows_z = top.rows(2);
    /// assert_eq!(rows_z.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [1.0, 5.0]);
    /// assert_eq!(rows_z.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [2.0, 6.0]);
    /// assert!(rows_z.next().is_none());
    /// ```
    pub fn rows(&self, dim: usize) -> Rows<'_, T> {
        let start_idx = flatten_idx(self.block_dims, self.block_strides, &self.dim_offsets);

        let (divisor, stride, row_strides) = with_alloca_zeroed(
            self.dims.len() * 2 * std::mem::size_of::<usize>(),
            |alloc| {
                let (tmp_idx, row_starts) = unsafe {
                    let (tmp_alloc, row_starts_alloc) = alloc.split_at_mut(alloc.len() / 2);
                    let tmp_idx = std::slice::from_raw_parts_mut(
                        tmp_alloc.as_mut_ptr() as *mut usize,
                        self.dims.len(),
                    );
                    let row_starts = std::slice::from_raw_parts_mut(
                        row_starts_alloc.as_mut_ptr() as *mut usize,
                        self.dims.len(),
                    );
                    (tmp_idx, row_starts)
                };

                tmp_idx[dim] = 1;
                let stride = unsafe {
                    flatten_idx_with_offset_unchecked(
                        self.block_strides,
                        tmp_idx,
                        &self.dim_offsets,
                    ) - start_idx
                };
                tmp_idx[dim] = 0;

                let mut divisor = 1;
                let mut row_start_idx = start_idx;
                let mut row_strides = Vec::with_capacity(self.dims.len() - 1);
                for (i, &d) in self.dims.iter().enumerate() {
                    if i != dim {
                        tmp_idx[i] = 1;
                        let row_end = unsafe {
                            flatten_idx_with_offset_unchecked(
                                self.block_strides,
                                tmp_idx,
                                &self.dim_offsets,
                            )
                        };
                        tmp_idx[i] = 0;

                        row_starts[i] = d - 1;
                        let next_row_start_idx = unsafe {
                            flatten_idx_with_offset_unchecked(
                                self.block_strides,
                                row_starts,
                                &self.dim_offsets,
                            )
                        };

                        let row_stride = row_end - row_start_idx;
                        row_strides.push((divisor, row_stride));

                        divisor *= d;
                        row_start_idx = next_row_start_idx;
                    }
                }

                (divisor, stride, row_strides)
            },
        );

        let num_rows = divisor;
        Rows {
            idx: 0,
            stride,
            num_rows,
            row_strides,
            row_idx: start_idx,
            row_len: self.dims[dim],
            data: self.block_data,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: Copy> VolumeWindow<'a, T> {
    /// Copies the contents of a window into another window.
    /// The two windows must be identical is size.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut block = VolumeBlock::new_with_data(&dims, data).unwrap();
    /// let mut window = block.window_mut();
    /// let (left, mut right) = window.split_into(0);
    /// let left = left.window();
    ///
    /// assert_eq!(left[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(left[[0usize, 1usize].as_ref()], 3.0);
    ///
    /// assert_eq!(right[[0usize, 0usize].as_ref()], 2.0);
    /// assert_eq!(right[[0usize, 1usize].as_ref()], 4.0);
    ///
    /// left.copy_to(&mut right);
    ///
    /// assert_eq!(left[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(left[[0usize, 1usize].as_ref()], 3.0);
    ///
    /// assert_eq!(right[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(right[[0usize, 1usize].as_ref()], 3.0);
    pub fn copy_to(&self, window: &mut VolumeWindowMut<'_, T>) {
        assert!(self.dims == window.dims);
        let src_rows = self.rows(0);
        let dst_rows = window.rows_mut(0);

        for (src, mut dst) in src_rows.zip(dst_rows) {
            let src = src.as_slice().unwrap();
            let dst = dst.as_slice_mut().unwrap();
            dst.copy_from_slice(src)
        }
    }
}

impl<'a, T: Clone> VolumeWindow<'a, T> {
    /// Clones the contents of a window into another window.
    /// The two windows must be identical is size.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut block = VolumeBlock::new_with_data(&dims, data).unwrap();
    /// let mut window = block.window_mut();
    /// let (left, mut right) = window.split_into(0);
    /// let left = left.window();
    ///
    /// assert_eq!(left[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(left[[0usize, 1usize].as_ref()], 3.0);
    ///
    /// assert_eq!(right[[0usize, 0usize].as_ref()], 2.0);
    /// assert_eq!(right[[0usize, 1usize].as_ref()], 4.0);
    ///
    /// left.clone_to(&mut right);
    ///
    /// assert_eq!(left[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(left[[0usize, 1usize].as_ref()], 3.0);
    ///
    /// assert_eq!(right[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(right[[0usize, 1usize].as_ref()], 3.0);
    pub fn clone_to(&self, window: &mut VolumeWindowMut<'_, T>) {
        assert!(self.dims == window.dims);
        let src_rows = self.rows(0);
        let dst_rows = window.rows_mut(0);

        for (src, mut dst) in src_rows.zip(dst_rows) {
            let src = src.as_slice().unwrap();
            let dst = dst.as_slice_mut().unwrap();
            dst.clone_from_slice(src)
        }
    }
}

unsafe impl<'a, T> Send for VolumeWindow<'a, T> where &'a [T]: Send {}
unsafe impl<'a, T> Sync for VolumeWindow<'a, T> where &'a [T]: Sync {}

impl<'a, T> Index<&[usize]> for VolumeWindow<'a, T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        let idx = flatten_idx_with_offset(
            self.block_dims,
            self.block_strides,
            index,
            &self.dim_offsets,
        );
        unsafe { &*self.block_data.get_unchecked(idx) }
    }
}

/// A borrowed mutable window into a [`VolumeBlock`].
#[derive(Debug)]
pub struct VolumeWindowMut<'a, T> {
    dims: Vec<usize>,
    dim_offsets: Vec<usize>,
    block_data: *mut [T],
    block_dims: &'a [usize],
    block_strides: &'a [usize],
    _phantom: PhantomData<&'a mut [T]>,
}

impl<'a, T> VolumeWindowMut<'a, T> {
    /// Constructs a shared window from the mutable window.
    pub fn window(&self) -> VolumeWindow<'_, T> {
        VolumeWindow {
            dims: self.dims.clone(),
            dim_offsets: self.dim_offsets.clone(),
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        }
    }

    /// Splits a window into two non overlapping halves.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut block = VolumeBlock::new_with_data(&dims, data).unwrap();
    /// let window = block.window_mut();
    /// let (left, right) = window.split(0);
    ///
    /// assert_eq!(left[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(left[[0usize, 1usize].as_ref()], 3.0);
    ///
    /// assert_eq!(right[[0usize, 0usize].as_ref()], 2.0);
    /// assert_eq!(right[[0usize, 1usize].as_ref()], 4.0);
    ///
    /// let (top, bottom) = window.split(1);
    ///
    /// assert_eq!(top[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(top[[1usize, 0usize].as_ref()], 2.0);
    ///
    /// assert_eq!(bottom[[0usize, 0usize].as_ref()], 3.0);
    /// assert_eq!(bottom[[1usize, 0usize].as_ref()], 4.0);
    /// ```
    pub fn split(&self, dim: usize) -> (VolumeWindow<'_, T>, VolumeWindow<'_, T>) {
        assert!(self.dims[dim] % 2 == 0);
        let mut dims = self.dims.clone();
        let offsets_left = self.dim_offsets.clone();
        let mut offsets_right = offsets_left.clone();

        dims[dim] /= 2;
        offsets_right[dim] += dims[dim];

        let left = VolumeWindow {
            dims: dims.clone(),
            dim_offsets: offsets_left,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        };

        let right = VolumeWindow {
            dims,
            dim_offsets: offsets_right,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        };

        (left, right)
    }

    /// Splits a window into two non overlapping halves.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut block = VolumeBlock::new_with_data(&dims, data).unwrap();
    /// let mut window = block.window_mut();
    /// let (left, right) = window.split_mut(0);
    ///
    /// assert_eq!(left[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(left[[0usize, 1usize].as_ref()], 3.0);
    ///
    /// assert_eq!(right[[0usize, 0usize].as_ref()], 2.0);
    /// assert_eq!(right[[0usize, 1usize].as_ref()], 4.0);
    ///
    /// let (top, bottom) = window.split_mut(1);
    ///
    /// assert_eq!(top[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(top[[1usize, 0usize].as_ref()], 2.0);
    ///
    /// assert_eq!(bottom[[0usize, 0usize].as_ref()], 3.0);
    /// assert_eq!(bottom[[1usize, 0usize].as_ref()], 4.0);
    /// ```
    pub fn split_mut(&mut self, dim: usize) -> (VolumeWindowMut<'_, T>, VolumeWindowMut<'_, T>) {
        assert!(self.dims[dim] % 2 == 0);
        let mut dims = self.dims.clone();
        let offsets_left = self.dim_offsets.clone();
        let mut offsets_right = offsets_left.clone();

        dims[dim] /= 2;
        offsets_right[dim] += dims[dim];

        let left = Self {
            dims: dims.clone(),
            dim_offsets: offsets_left,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        };

        let right = Self {
            dims,
            dim_offsets: offsets_right,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        };

        (left, right)
    }

    /// Splits a window into two non overlapping halves.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut block = VolumeBlock::new_with_data(&dims, data).unwrap();
    /// let mut window = block.window_mut();
    /// let (left, right) = window.split_into(0);
    ///
    /// assert_eq!(left[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(left[[0usize, 1usize].as_ref()], 3.0);
    ///
    /// assert_eq!(right[[0usize, 0usize].as_ref()], 2.0);
    /// assert_eq!(right[[0usize, 1usize].as_ref()], 4.0);
    ///
    /// let mut window = block.window_mut();
    /// let (top, bottom) = window.split_into(1);
    ///
    /// assert_eq!(top[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(top[[1usize, 0usize].as_ref()], 2.0);
    ///
    /// assert_eq!(bottom[[0usize, 0usize].as_ref()], 3.0);
    /// assert_eq!(bottom[[1usize, 0usize].as_ref()], 4.0);
    /// ```
    pub fn split_into(self, dim: usize) -> (VolumeWindowMut<'a, T>, VolumeWindowMut<'a, T>) {
        assert!(self.dims[dim] % 2 == 0);
        let mut dims = self.dims;
        let offsets_left = self.dim_offsets;
        let mut offsets_right = offsets_left.clone();

        dims[dim] /= 2;
        offsets_right[dim] += dims[dim];

        let left = Self {
            dims: dims.clone(),
            dim_offsets: offsets_left,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        };

        let right = Self {
            dims,
            dim_offsets: offsets_right,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        };

        (left, right)
    }

    /// Borrows a subwindow in the range `[0..dims[0], ...]` from the window.
    pub fn custom_range(&self, range: &[usize]) -> VolumeWindow<'_, T> {
        assert_eq!(self.dims.len(), range.len());
        assert!(range
            .iter()
            .zip(&self.dim_offsets)
            .zip(&self.dims)
            .all(|((&r, &o), &d)| r + o <= d));

        let dim_offsets = self.dim_offsets.clone();
        VolumeWindow {
            dims: range.into(),
            dim_offsets,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        }
    }

    /// Borrows a mutable subwindow in the range `[0..dims[0], ...]` from the window.
    pub fn custom_range_mut(&mut self, range: &[usize]) -> VolumeWindowMut<'_, T> {
        assert_eq!(self.dims.len(), range.len());
        assert!(range
            .iter()
            .zip(&self.dim_offsets)
            .zip(&self.dims)
            .all(|((&r, &o), &d)| r + o <= d));

        let dim_offsets = self.dim_offsets.clone();
        Self {
            dims: range.into(),
            dim_offsets,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        }
    }

    /// Borrows a subwindow in the range `[offset[0]..dims[0], ...]` from the window.
    pub fn custom_window(&self, offset: &[usize], dims: &[usize]) -> VolumeWindow<'_, T> {
        assert_eq!(self.dims.len(), offset.len());
        assert_eq!(self.dims.len(), dims.len());
        assert!(offset
            .iter()
            .zip(dims)
            .zip(&self.dim_offsets)
            .zip(&self.dims)
            .all(|(((&n_o, &n_d), &o), &d)| n_o + n_d + o <= d));

        let dim_offsets = offset
            .iter()
            .zip(&self.dim_offsets)
            .map(|(&n_o, &o)| n_o + o)
            .collect();
        VolumeWindow {
            dims: dims.into(),
            dim_offsets,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        }
    }

    /// Borrows a mutable subwindow in the range `[offset[0]..dims[0], ...]` from the window.
    pub fn custom_window_mut(
        &mut self,
        offset: &[usize],
        dims: &[usize],
    ) -> VolumeWindowMut<'_, T> {
        assert_eq!(self.dims.len(), offset.len());
        assert_eq!(self.dims.len(), dims.len());
        assert!(offset
            .iter()
            .zip(dims)
            .zip(&self.dim_offsets)
            .zip(&self.dims)
            .all(|(((&n_o, &n_d), &o), &d)| n_o + n_d + o <= d));

        let dim_offsets = offset
            .iter()
            .zip(&self.dim_offsets)
            .map(|(&n_o, &o)| n_o + o)
            .collect();
        Self {
            dims: dims.into(),
            dim_offsets,
            block_data: self.block_data,
            block_dims: self.block_dims,
            block_strides: self.block_strides,
            _phantom: PhantomData,
        }
    }

    /// Returns the dimensions of the window.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Constructs an iterator over the rows of a window.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2, 2];
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let mut block = VolumeBlock::new_with_data(&dims, data).unwrap();
    /// let window = block.window_mut();
    ///
    /// let mut rows_x = window.rows(0);
    /// assert_eq!(rows_x.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [1.0, 2.0]);
    /// assert_eq!(rows_x.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [3.0, 4.0]);
    /// assert_eq!(rows_x.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [5.0, 6.0]);
    /// assert_eq!(rows_x.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [7.0, 8.0]);
    /// assert!(rows_x.next().is_none());
    ///
    /// let mut rows_y = window.rows(1);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [1.0, 3.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [2.0, 4.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [5.0, 7.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [6.0, 8.0]);
    /// assert!(rows_y.next().is_none());
    ///
    /// let mut rows_z = window.rows(2);
    /// assert_eq!(rows_z.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [1.0, 5.0]);
    /// assert_eq!(rows_z.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [2.0, 6.0]);
    /// assert_eq!(rows_z.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [3.0, 7.0]);
    /// assert_eq!(rows_z.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [4.0, 8.0]);
    /// assert!(rows_z.next().is_none());
    ///
    /// let (top, _) = window.split(1);
    /// let mut rows_x = top.rows(0);
    /// assert_eq!(rows_x.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [1.0, 2.0]);
    /// assert_eq!(rows_x.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [5.0, 6.0]);
    /// assert!(rows_x.next().is_none());
    ///
    /// let mut rows_y = top.rows(1);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [1.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [2.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [5.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [6.0]);
    /// assert!(rows_y.next().is_none());
    ///
    /// let mut rows_z = top.rows(2);
    /// assert_eq!(rows_z.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [1.0, 5.0]);
    /// assert_eq!(rows_z.next().unwrap().iter().cloned().collect::<Vec<f32>>(), [2.0, 6.0]);
    /// assert!(rows_z.next().is_none());
    /// ```
    pub fn rows(&self, dim: usize) -> Rows<'_, T> {
        let start_idx = flatten_idx(self.block_dims, self.block_strides, &self.dim_offsets);

        let (divisor, stride, row_strides) = with_alloca_zeroed(
            self.dims.len() * 2 * std::mem::size_of::<usize>(),
            |alloc| {
                let (tmp_idx, row_starts) = unsafe {
                    let (tmp_alloc, row_starts_alloc) = alloc.split_at_mut(alloc.len() / 2);
                    let tmp_idx = std::slice::from_raw_parts_mut(
                        tmp_alloc.as_mut_ptr() as *mut usize,
                        self.dims.len(),
                    );
                    let row_starts = std::slice::from_raw_parts_mut(
                        row_starts_alloc.as_mut_ptr() as *mut usize,
                        self.dims.len(),
                    );
                    (tmp_idx, row_starts)
                };

                tmp_idx[dim] = 1;
                let stride = unsafe {
                    flatten_idx_with_offset_unchecked(
                        self.block_strides,
                        tmp_idx,
                        &self.dim_offsets,
                    ) - start_idx
                };
                tmp_idx[dim] = 0;

                let mut divisor = 1;
                let mut row_start_idx = start_idx;
                let mut row_strides = Vec::with_capacity(self.dims.len() - 1);
                for (i, &d) in self.dims.iter().enumerate() {
                    if i != dim {
                        tmp_idx[i] = 1;
                        let row_end = unsafe {
                            flatten_idx_with_offset_unchecked(
                                self.block_strides,
                                tmp_idx,
                                &self.dim_offsets,
                            )
                        };
                        tmp_idx[i] = 0;

                        row_starts[i] = d - 1;
                        let next_row_start_idx = unsafe {
                            flatten_idx_with_offset_unchecked(
                                self.block_strides,
                                row_starts,
                                &self.dim_offsets,
                            )
                        };

                        let row_stride = row_end - row_start_idx;
                        row_strides.push((divisor, row_stride));

                        divisor *= d;
                        row_start_idx = next_row_start_idx;
                    }
                }

                (divisor, stride, row_strides)
            },
        );

        let num_rows = divisor;
        Rows {
            idx: 0,
            stride,
            num_rows,
            row_strides,
            row_idx: start_idx,
            row_len: self.dims[dim],
            data: self.block_data,
            _phantom: PhantomData,
        }
    }

    /// Constructs a mutable iterator over the rows of a window.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2, 2];
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let mut block = VolumeBlock::new_with_data(&dims, data).unwrap();
    /// let mut window = block.window_mut();
    ///
    /// let mut rows_x = window.rows_mut(0);
    /// assert_eq!(rows_x.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [1.0, 2.0]);
    /// assert_eq!(rows_x.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [3.0, 4.0]);
    /// assert_eq!(rows_x.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [5.0, 6.0]);
    /// assert_eq!(rows_x.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [7.0, 8.0]);
    /// assert!(rows_x.next().is_none());
    ///
    /// let mut rows_y = window.rows_mut(1);
    /// assert_eq!(rows_y.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [1.0, 3.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [2.0, 4.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [5.0, 7.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [6.0, 8.0]);
    /// assert!(rows_y.next().is_none());
    ///
    /// let mut rows_z = window.rows_mut(2);
    /// assert_eq!(rows_z.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [1.0, 5.0]);
    /// assert_eq!(rows_z.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [2.0, 6.0]);
    /// assert_eq!(rows_z.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [3.0, 7.0]);
    /// assert_eq!(rows_z.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [4.0, 8.0]);
    /// assert!(rows_z.next().is_none());
    ///
    /// let (mut top, _) = window.split_mut(1);
    /// let mut rows_x = top.rows_mut(0);
    /// assert_eq!(rows_x.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [1.0, 2.0]);
    /// assert_eq!(rows_x.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [5.0, 6.0]);
    /// assert!(rows_x.next().is_none());
    ///
    /// let mut rows_y = top.rows_mut(1);
    /// assert_eq!(rows_y.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [1.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [2.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [5.0]);
    /// assert_eq!(rows_y.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [6.0]);
    /// assert!(rows_y.next().is_none());
    ///
    /// let mut rows_z = top.rows_mut(2);
    /// assert_eq!(rows_z.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [1.0, 5.0]);
    /// assert_eq!(rows_z.next().unwrap().iter().map(|x| *x).collect::<Vec<f32>>(), [2.0, 6.0]);
    /// assert!(rows_z.next().is_none());
    /// ```
    pub fn rows_mut(&mut self, dim: usize) -> RowsMut<'_, T> {
        let start_idx = flatten_idx(self.block_dims, self.block_strides, &self.dim_offsets);

        let (divisor, stride, row_strides) = with_alloca_zeroed(
            self.dims.len() * 2 * std::mem::size_of::<usize>(),
            |alloc| {
                let (tmp_idx, row_starts) = unsafe {
                    let (tmp_alloc, row_starts_alloc) = alloc.split_at_mut(alloc.len() / 2);
                    let tmp_idx = std::slice::from_raw_parts_mut(
                        tmp_alloc.as_mut_ptr() as *mut usize,
                        self.dims.len(),
                    );
                    let row_starts = std::slice::from_raw_parts_mut(
                        row_starts_alloc.as_mut_ptr() as *mut usize,
                        self.dims.len(),
                    );
                    (tmp_idx, row_starts)
                };

                tmp_idx[dim] = 1;
                let stride = unsafe {
                    flatten_idx_with_offset_unchecked(
                        self.block_strides,
                        tmp_idx,
                        &self.dim_offsets,
                    ) - start_idx
                };
                tmp_idx[dim] = 0;

                let mut divisor = 1;
                let mut row_start_idx = start_idx;
                let mut row_strides = Vec::with_capacity(self.dims.len() - 1);
                for (i, &d) in self.dims.iter().enumerate() {
                    if i != dim {
                        tmp_idx[i] = 1;
                        let row_end = unsafe {
                            flatten_idx_with_offset_unchecked(
                                self.block_strides,
                                tmp_idx,
                                &self.dim_offsets,
                            )
                        };
                        tmp_idx[i] = 0;

                        row_starts[i] = d - 1;
                        let next_row_start_idx = unsafe {
                            flatten_idx_with_offset_unchecked(
                                self.block_strides,
                                row_starts,
                                &self.dim_offsets,
                            )
                        };

                        let row_stride = row_end - row_start_idx;
                        row_strides.push((divisor, row_stride));

                        divisor *= d;
                        row_start_idx = next_row_start_idx;
                    }
                }

                (divisor, stride, row_strides)
            },
        );

        let num_rows = divisor;
        RowsMut {
            idx: 0,
            stride,
            num_rows,
            row_strides,
            row_idx: start_idx,
            row_len: self.dims[dim],
            data: self.block_data,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: Copy> VolumeWindowMut<'a, T> {
    /// Copies the contents of a window into another window.
    /// The two windows must be identical is size.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut block = VolumeBlock::new_with_data(&dims, data).unwrap();
    /// let mut window = block.window_mut();
    /// let (left, mut right) = window.split_into(0);
    ///
    /// assert_eq!(left[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(left[[0usize, 1usize].as_ref()], 3.0);
    ///
    /// assert_eq!(right[[0usize, 0usize].as_ref()], 2.0);
    /// assert_eq!(right[[0usize, 1usize].as_ref()], 4.0);
    ///
    /// left.copy_to(&mut right);
    ///
    /// assert_eq!(left[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(left[[0usize, 1usize].as_ref()], 3.0);
    ///
    /// assert_eq!(right[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(right[[0usize, 1usize].as_ref()], 3.0);
    pub fn copy_to(&self, window: &mut VolumeWindowMut<'_, T>) {
        assert!(self.dims == window.dims);
        let src_rows = self.rows(0);
        let dst_rows = window.rows_mut(0);

        for (src, mut dst) in src_rows.zip(dst_rows) {
            let src = src.as_slice().unwrap();
            let dst = dst.as_slice_mut().unwrap();
            dst.copy_from_slice(src)
        }
    }
}

impl<'a, T: Clone> VolumeWindowMut<'a, T> {
    /// Clones the contents of a window into another window.
    /// The two windows must be identical is size.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut block = VolumeBlock::new_with_data(&dims, data).unwrap();
    /// let mut window = block.window_mut();
    /// let (left, mut right) = window.split_into(0);
    ///
    /// assert_eq!(left[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(left[[0usize, 1usize].as_ref()], 3.0);
    ///
    /// assert_eq!(right[[0usize, 0usize].as_ref()], 2.0);
    /// assert_eq!(right[[0usize, 1usize].as_ref()], 4.0);
    ///
    /// left.clone_to(&mut right);
    ///
    /// assert_eq!(left[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(left[[0usize, 1usize].as_ref()], 3.0);
    ///
    /// assert_eq!(right[[0usize, 0usize].as_ref()], 1.0);
    /// assert_eq!(right[[0usize, 1usize].as_ref()], 3.0);
    pub fn clone_to(&self, window: &mut VolumeWindowMut<'_, T>) {
        assert!(self.dims == window.dims);
        let src_rows = self.rows(0);
        let dst_rows = window.rows_mut(0);

        for (src, mut dst) in src_rows.zip(dst_rows) {
            let src = src.as_slice().unwrap();
            let dst = dst.as_slice_mut().unwrap();
            dst.clone_from_slice(src)
        }
    }
}

unsafe impl<'a, T> Send for VolumeWindowMut<'a, T> where &'a mut [T]: Send {}
unsafe impl<'a, T> Sync for VolumeWindowMut<'a, T> where &'a mut [T]: Sync {}

impl<'a, T> Index<&[usize]> for VolumeWindowMut<'a, T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        let idx = flatten_idx_with_offset(
            self.block_dims,
            self.block_strides,
            index,
            &self.dim_offsets,
        );
        unsafe { &*self.block_data.get_unchecked_mut(idx) }
    }
}

impl<'a, T> IndexMut<&[usize]> for VolumeWindowMut<'a, T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        let idx = flatten_idx_with_offset(
            self.block_dims,
            self.block_strides,
            index,
            &self.dim_offsets,
        );
        unsafe { &mut *self.block_data.get_unchecked_mut(idx) }
    }
}

/// An iterator over the lanes of a [`VolumeWindow`] or [`VolumeWindowMut`].
#[derive(Debug)]
pub struct Rows<'a, T> {
    idx: usize,
    stride: usize,
    row_len: usize,
    row_idx: usize,
    num_rows: usize,
    row_strides: Vec<(usize, usize)>,
    data: *const [T],
    _phantom: PhantomData<&'a [T]>,
}

impl<'a, T: 'a> Iterator for Rows<'a, T> {
    type Item = Row<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.num_rows {
            self.idx += 1;

            let row_start = self.row_idx;
            let &(_, row_stride) = self
                .row_strides
                .iter()
                .rev()
                .find(|(div, _)| self.idx % div == 0)
                .unwrap_or(&(0, 0));
            self.row_idx += row_stride;

            Some(Row {
                stride: self.stride,
                row_len: self.row_len,
                row_start,
                data: self.data,
                _phantom: PhantomData,
            })
        } else {
            None
        }
    }
}

unsafe impl<'a, T> Send for Rows<'a, T> where &'a [T]: Send {}
unsafe impl<'a, T> Sync for Rows<'a, T> where &'a [T]: Sync {}

/// A mutable iterator over the lanes of a [`VolumeWindowMut`].
#[derive(Debug)]
pub struct RowsMut<'a, T> {
    idx: usize,
    stride: usize,
    row_len: usize,
    row_idx: usize,
    num_rows: usize,
    row_strides: Vec<(usize, usize)>,
    data: *mut [T],
    _phantom: PhantomData<&'a mut [T]>,
}

impl<'a, T: 'a> Iterator for RowsMut<'a, T> {
    type Item = RowMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.num_rows {
            self.idx += 1;

            let row_start = self.row_idx;
            let &(_, row_stride) = self
                .row_strides
                .iter()
                .rev()
                .find(|(div, _)| self.idx % div == 0)
                .unwrap_or(&(0, 0));
            self.row_idx += row_stride;

            Some(RowMut {
                stride: self.stride,
                row_len: self.row_len,
                row_start,
                data: self.data,
                _phantom: PhantomData,
            })
        } else {
            None
        }
    }
}

unsafe impl<'a, T> Send for RowsMut<'a, T> where &'a mut [T]: Send {}
unsafe impl<'a, T> Sync for RowsMut<'a, T> where &'a mut [T]: Sync {}

/// Information regarding a lane of a [`VolumeWindow`] or [`VolumeWindowMut`].
#[derive(Debug)]
pub struct Row<'a, T> {
    stride: usize,
    row_len: usize,
    row_start: usize,
    data: *const [T],
    _phantom: PhantomData<&'a [T]>,
}

impl<'a, T> Row<'a, T> {
    /// Checks if the lane contains any elements.
    pub fn is_empty(&self) -> bool {
        self.row_len == 0
    }

    /// Returns the number of elements contained in the lane.
    pub fn len(&self) -> usize {
        self.row_len
    }

    /// Returns a slice to the lane if all it's elements are
    /// stored contiguously in memory.
    pub fn as_slice(&self) -> Option<&[T]> {
        if self.stride == 1 {
            unsafe {
                let slice = self
                    .data
                    .get_unchecked(self.row_start..self.row_start + self.row_len);
                Some(&*slice)
            }
        } else {
            None
        }
    }

    /// Fetches a reference to an element without any bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`.
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        let flat_idx = self.row_start + (index * self.stride);
        &*self.data.get_unchecked(flat_idx)
    }

    /// Constructs an iterator over the elements of the lane.
    pub fn iter(&self) -> RowIter<'_, 'a, T> {
        self.into_iter()
    }
}

impl<'a, T> Index<usize> for Row<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.row_len);
        unsafe { self.get_unchecked(index) }
    }
}

impl<'a, 'b, T> IntoIterator for &'a Row<'b, T> {
    type Item = &'a T;

    type IntoIter = RowIter<'a, 'b, T>;

    fn into_iter(self) -> Self::IntoIter {
        RowIter { idx: 0, row: self }
    }
}

unsafe impl<'a, T> Send for Row<'a, T> where &'a [T]: Send {}
unsafe impl<'a, T> Sync for Row<'a, T> where &'a [T]: Sync {}

/// Iterator over the elements of a [`Row`] or a [`RowMut`].
#[derive(Debug)]
pub struct RowIter<'a, 'b, T> {
    idx: usize,
    row: &'a Row<'b, T>,
}

impl<'a, 'b, T> Iterator for RowIter<'a, 'b, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.row.row_len {
            let idx = self.idx;
            self.idx += 1;

            let val = unsafe { self.row.get_unchecked(idx) };
            Some(val)
        } else {
            None
        }
    }
}

/// Information regarding a mutable lane of a [`VolumeWindowMut`].
#[derive(Debug)]
pub struct RowMut<'a, T> {
    stride: usize,
    row_len: usize,
    row_start: usize,
    data: *mut [T],
    _phantom: PhantomData<&'a mut [T]>,
}

impl<'a, T> RowMut<'a, T> {
    /// Checks if the lane contains any elements.
    pub fn is_empty(&self) -> bool {
        self.row_len == 0
    }

    /// Returns the number of elements contained in the lane.
    pub fn len(&self) -> usize {
        self.row_len
    }

    /// Returns a slice to the lane if all it's elements are
    /// stored contiguously in memory.
    pub fn as_slice(&self) -> Option<&[T]> {
        if self.stride == 1 {
            unsafe {
                let data = self.data as *const [T];
                let slice = data.get_unchecked(self.row_start..self.row_start + self.row_len);
                Some(&*slice)
            }
        } else {
            None
        }
    }

    /// Returns a mutable slice to the lane if all it's elements are
    /// stored contiguously in memory.
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        if self.stride == 1 {
            unsafe {
                let slice = self
                    .data
                    .get_unchecked_mut(self.row_start..self.row_start + self.row_len);
                Some(&mut *slice)
            }
        } else {
            None
        }
    }

    /// Fetches a reference to an element without any bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`.
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        let flat_idx = self.row_start + (index * self.stride);
        let data = self.data as *const [T];
        &*data.get_unchecked(flat_idx)
    }

    /// Fetches a mutable reference to an element without any bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`.
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        &mut *self.get_unchecked_mut_ptr(index)
    }

    unsafe fn get_unchecked_mut_ptr(&mut self, index: usize) -> *mut T {
        let flat_idx = self.row_start + (index * self.stride);
        self.data.get_unchecked_mut(flat_idx)
    }

    /// Borrows the lane immutably.
    pub fn as_row(&self) -> Row<'_, T> {
        Row {
            stride: self.stride,
            row_len: self.row_len,
            row_start: self.row_start,
            data: self.data,
            _phantom: PhantomData,
        }
    }

    /// Constructs a mutable iterator over the elements of the lane.
    pub fn iter_mut(&mut self) -> RowIterMut<'_, 'a, T> {
        self.into_iter()
    }
}

impl<'a, T> Index<usize> for RowMut<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.row_len);
        unsafe { self.get_unchecked(index) }
    }
}

impl<'a, T> IndexMut<usize> for RowMut<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.row_len);
        unsafe { self.get_unchecked_mut(index) }
    }
}

impl<'a, 'b, T> IntoIterator for &'a mut RowMut<'b, T> {
    type Item = &'a mut T;

    type IntoIter = RowIterMut<'a, 'b, T>;

    fn into_iter(self) -> Self::IntoIter {
        RowIterMut { idx: 0, row: self }
    }
}

unsafe impl<'a, T> Send for RowMut<'a, T> where &'a mut [T]: Send {}
unsafe impl<'a, T> Sync for RowMut<'a, T> where &'a mut [T]: Sync {}

/// Iterator over the elements of a [`RowMut`].
#[derive(Debug)]
pub struct RowIterMut<'a, 'b, T> {
    idx: usize,
    row: &'a mut RowMut<'b, T>,
}

impl<'a, 'b, T> Iterator for RowIterMut<'a, 'b, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.row.row_len {
            let idx = self.idx;
            self.idx += 1;

            let val = unsafe { &mut *self.row.get_unchecked_mut_ptr(idx) };
            Some(val)
        } else {
            None
        }
    }
}
