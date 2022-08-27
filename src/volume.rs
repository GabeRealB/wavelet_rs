//! Volume utilities.

use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct VolumeBlock {
    data: Vec<f32>,
    dims: Vec<usize>,
}

impl VolumeBlock {
    /// Constructs a new block.
    ///
    /// # Examples
    ///
    /// ```
    /// use wavelet_rs::volume::VolumeBlock;
    ///
    /// let dims = [2, 2];
    /// let block = VolumeBlock::new(&dims).unwrap();
    ///
    /// assert_eq!(block[[0usize, 0usize].as_ref()], 0.0);
    /// assert_eq!(block[[1usize, 0usize].as_ref()], 0.0);
    /// assert_eq!(block[[0usize, 1usize].as_ref()], 0.0);
    /// assert_eq!(block[[1usize, 1usize].as_ref()], 0.0);
    /// ```
    pub fn new(dims: &[usize]) -> Result<Self, VolumeError> {
        let num_elements = dims.iter().product();
        let data = vec![0.0; num_elements];
        Self::new_with_data(dims, data)
    }

    /// Constructs a new block.
    ///
    /// # Examples
    ///
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
    pub fn new_with_data(dims: &[usize], data: Vec<f32>) -> Result<Self, VolumeError> {
        let num_elements = dims.iter().product();

        if dims.is_empty() {
            Err(VolumeError::ZeroBlockLength)
        } else if let Some(len) = dims.iter().find(|&&len| len == 0 || !len.is_power_of_two()) {
            Err(VolumeError::InvalidDimensionLenght { length: *len })
        } else if num_elements != data.len() {
            Err(VolumeError::InvalidNumberOfElements {
                got: data.len(),
                required: num_elements,
            })
        } else {
            Ok(Self {
                data,
                dims: dims.into(),
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
    pub fn window(&self) -> VolumeWindow<'_> {
        VolumeWindow {
            block: self,
            dims: self.dims.clone(),
            dim_offsets: vec![0; self.dims.len()],
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
    pub fn window_mut(&mut self) -> VolumeWindowMut<'_> {
        VolumeWindowMut {
            block: self,
            dims: self.dims.clone(),
            dim_offsets: vec![0; self.dims.len()],
            _phantom: PhantomData,
        }
    }

    /// Checks that the data contained inside the volumes is equal
    /// apart from a specified error value.
    pub fn is_equal(&self, other: &Self, eps: f32) -> bool {
        self.dims == other.dims
            && self
                .data
                .iter()
                .zip(&other.data)
                .all(|(this, other)| (*this - *other).abs() < eps)
    }

    /// Returns a slice to the flat representation of the volume.
    pub fn flatten(&self) -> &[f32] {
        &self.data
    }

    /// Returns a mutable slice to the flat representation of the volume.
    pub fn flatten_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    fn flatten_idx(&self, index: &[usize]) -> usize {
        assert!(index.len() == self.dims.len());

        if index
            .iter()
            .zip(self.dims.iter())
            .any(|(&idx, &dim_len)| idx > dim_len)
        {
            panic!("out of bounds");
        }

        let mut idx = index[0];
        let mut offset_multiplier = 1;
        for (i, &dim_idx) in index.iter().enumerate().skip(1) {
            offset_multiplier *= self.dims[i - 1];
            idx += offset_multiplier * dim_idx;
        }

        idx
    }

    fn flatten_idx_with_offset(&self, index: &[usize], offset: &[usize]) -> usize {
        assert!(offset.len() == self.dims.len());
        assert!(index.len() == self.dims.len());

        if index
            .iter()
            .zip(offset.iter())
            .zip(self.dims.iter())
            .any(|((&idx, &offset), &dim_len)| (idx + offset) > dim_len)
        {
            panic!("out of bounds");
        }

        let mut idx = index[0] + offset[0];
        let mut offset_multiplier = 1;
        for (i, (&dim_idx, &offset)) in index.iter().zip(offset.iter()).enumerate().skip(1) {
            offset_multiplier *= self.dims[i - 1];
            idx += offset_multiplier * (dim_idx + offset);
        }

        idx
    }

    fn index_offset(&self, index: &[usize], offset: &[usize]) -> &f32 {
        let idx = self.flatten_idx_with_offset(index, offset);
        &self.data[idx]
    }

    fn index_offset_mut(&mut self, index: &[usize], offset: &[usize]) -> &mut f32 {
        let idx = self.flatten_idx_with_offset(index, offset);
        &mut self.data[idx]
    }

    fn sub_slice(&self, start: usize, len: usize) -> &[f32] {
        &self.data[start..start + len]
    }

    fn sub_slice_mut(&mut self, start: usize, len: usize) -> &mut [f32] {
        &mut self.data[start..start + len]
    }
}

impl Index<usize> for VolumeBlock {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for VolumeBlock {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Index<&[usize]> for VolumeBlock {
    type Output = f32;

    fn index(&self, index: &[usize]) -> &Self::Output {
        let idx = self.flatten_idx(index);
        &self[idx]
    }
}

impl IndexMut<&[usize]> for VolumeBlock {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        let idx = self.flatten_idx(index);
        &mut self[idx]
    }
}

#[derive(Error, Debug)]
pub enum VolumeError {
    #[error("a block length of 0 is not supported")]
    ZeroBlockLength,
    #[error("invalid length for dimension (got {length})")]
    InvalidDimensionLenght { length: usize },
    #[error("invalid number of elements (got {got}, required {required})")]
    InvalidNumberOfElements { got: usize, required: usize },
}

#[derive(Debug, Clone)]
pub struct VolumeWindow<'a> {
    dims: Vec<usize>,
    dim_offsets: Vec<usize>,
    block: *const VolumeBlock,
    _phantom: PhantomData<&'a VolumeBlock>,
}

impl<'a> VolumeWindow<'a> {
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
    pub fn split(&self, dim: usize) -> (VolumeWindow<'_>, VolumeWindow<'_>) {
        assert!(self.dims[dim] > 1);
        let mut dims = self.dims.clone();
        let offsets_left = self.dim_offsets.clone();
        let mut offsets_right = offsets_left.clone();

        dims[dim] /= 2;
        offsets_right[dim] += dims[dim];

        let left = Self {
            block: self.block,
            dims: dims.clone(),
            dim_offsets: offsets_left,
            _phantom: PhantomData,
        };

        let right = Self {
            block: self.block,
            dims,
            dim_offsets: offsets_right,
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
    pub fn split_into(self, dim: usize) -> (VolumeWindow<'a>, VolumeWindow<'a>) {
        assert!(self.dims[dim] > 1);
        let mut dims = self.dims;
        let offsets_left = self.dim_offsets;
        let mut offsets_right = offsets_left.clone();

        dims[dim] /= 2;
        offsets_right[dim] += dims[dim];

        let left = Self {
            block: self.block,
            dims: dims.clone(),
            dim_offsets: offsets_left,
            _phantom: PhantomData,
        };

        let right = Self {
            block: self.block,
            dims,
            dim_offsets: offsets_right,
            _phantom: PhantomData,
        };

        (left, right)
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
    pub fn rows(&self, dim: usize) -> Rows<'_, 'a> {
        let num_rows =
            self.dims.iter().enumerate().fold(
                1,
                |acc, (idx, &len)| {
                    if dim != idx {
                        acc * len
                    } else {
                        acc
                    }
                },
            );

        let start_idx = unsafe { (*self.block).flatten_idx(&self.dim_offsets) };

        let mut tmp_idx = vec![0; self.dims.len()];
        tmp_idx[dim] = 1;

        let stride = unsafe {
            (*self.block).flatten_idx_with_offset(&tmp_idx, &self.dim_offsets) - start_idx
        };

        let row_idx = vec![0; self.dims.len()];
        Rows {
            dim,
            idx: 0,
            stride,
            row_len: self.dims[dim],
            num_rows,
            row_idx,
            block: self.block,
            dims: &self.dims,
            dim_offsets: &self.dim_offsets,
            _phantom: PhantomData,
        }
    }

    /// Copies a window into another window.
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
    pub fn copy_to(&self, window: &mut VolumeWindowMut<'_>) {
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

unsafe impl<'a> Send for VolumeWindow<'a> where &'a VolumeBlock: Send {}
unsafe impl<'a> Sync for VolumeWindow<'a> where &'a VolumeBlock: Sync {}

impl<'a> Index<&[usize]> for VolumeWindow<'a> {
    type Output = f32;

    fn index(&self, index: &[usize]) -> &Self::Output {
        unsafe { (*self.block).index_offset(index, &self.dim_offsets) }
    }
}

#[derive(Debug)]
pub struct VolumeWindowMut<'a> {
    dims: Vec<usize>,
    dim_offsets: Vec<usize>,
    block: *mut VolumeBlock,
    _phantom: PhantomData<&'a mut VolumeBlock>,
}

impl<'a> VolumeWindowMut<'a> {
    /// Constructs a shared window from the mutable window.
    pub fn window(&self) -> VolumeWindow {
        VolumeWindow {
            block: self.block,
            dims: self.dims.clone(),
            dim_offsets: self.dim_offsets.clone(),
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
    pub fn split(&self, dim: usize) -> (VolumeWindow<'_>, VolumeWindow<'_>) {
        assert!(self.dims[dim] > 1);
        let mut dims = self.dims.clone();
        let offsets_left = self.dim_offsets.clone();
        let mut offsets_right = offsets_left.clone();

        dims[dim] /= 2;
        offsets_right[dim] += dims[dim];

        let left = VolumeWindow {
            block: self.block,
            dims: dims.clone(),
            dim_offsets: offsets_left,
            _phantom: PhantomData,
        };

        let right = VolumeWindow {
            block: self.block,
            dims,
            dim_offsets: offsets_right,
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
    pub fn split_mut(&mut self, dim: usize) -> (VolumeWindowMut<'_>, VolumeWindowMut<'_>) {
        assert!(self.dims[dim] > 1);
        let mut dims = self.dims.clone();
        let offsets_left = self.dim_offsets.clone();
        let mut offsets_right = offsets_left.clone();

        dims[dim] /= 2;
        offsets_right[dim] += dims[dim];

        let left = Self {
            block: self.block,
            dims: dims.clone(),
            dim_offsets: offsets_left,
            _phantom: PhantomData,
        };

        let right = Self {
            block: self.block,
            dims,
            dim_offsets: offsets_right,
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
    pub fn split_into(self, dim: usize) -> (VolumeWindowMut<'a>, VolumeWindowMut<'a>) {
        assert!(self.dims[dim] > 1);
        let mut dims = self.dims;
        let offsets_left = self.dim_offsets;
        let mut offsets_right = offsets_left.clone();

        dims[dim] /= 2;
        offsets_right[dim] += dims[dim];

        let left = Self {
            block: self.block,
            dims: dims.clone(),
            dim_offsets: offsets_left,
            _phantom: PhantomData,
        };

        let right = Self {
            block: self.block,
            dims,
            dim_offsets: offsets_right,
            _phantom: PhantomData,
        };

        (left, right)
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
    pub fn rows(&self, dim: usize) -> Rows<'_, 'a> {
        let num_rows =
            self.dims.iter().enumerate().fold(
                1,
                |acc, (idx, &len)| {
                    if dim != idx {
                        acc * len
                    } else {
                        acc
                    }
                },
            );

        let start_idx = unsafe { (*self.block).flatten_idx(&self.dim_offsets) };

        let mut tmp_idx = vec![0; self.dims.len()];
        tmp_idx[dim] = 1;

        let stride = unsafe {
            (*self.block).flatten_idx_with_offset(&tmp_idx, &self.dim_offsets) - start_idx
        };

        let row_idx = vec![0; self.dims.len()];
        Rows {
            dim,
            idx: 0,
            stride,
            row_len: self.dims[dim],
            num_rows,
            row_idx,
            block: self.block,
            dims: &self.dims,
            dim_offsets: &self.dim_offsets,
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
    pub fn rows_mut(&mut self, dim: usize) -> RowsMut<'_, 'a> {
        let num_rows =
            self.dims.iter().enumerate().fold(
                1,
                |acc, (idx, &len)| {
                    if dim != idx {
                        acc * len
                    } else {
                        acc
                    }
                },
            );

        let start_idx = unsafe { (*self.block).flatten_idx(&self.dim_offsets) };

        let mut tmp_idx = vec![0; self.dims.len()];
        tmp_idx[dim] = 1;

        let stride = unsafe {
            (*self.block).flatten_idx_with_offset(&tmp_idx, &self.dim_offsets) - start_idx
        };

        let row_idx = vec![0; self.dims.len()];
        RowsMut {
            dim,
            idx: 0,
            stride,
            row_len: self.dims[dim],
            num_rows,
            row_idx,
            block: self.block,
            dims: &self.dims,
            dim_offsets: &self.dim_offsets,
            _phantom: PhantomData,
        }
    }

    /// Copies a window into another window.
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
    pub fn copy_to(&self, window: &mut VolumeWindowMut<'_>) {
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

unsafe impl<'a> Send for VolumeWindowMut<'a> where &'a mut VolumeBlock: Send {}
unsafe impl<'a> Sync for VolumeWindowMut<'a> where &'a mut VolumeBlock: Sync {}

impl<'a> Index<&[usize]> for VolumeWindowMut<'a> {
    type Output = f32;

    fn index(&self, index: &[usize]) -> &Self::Output {
        unsafe { (*self.block).index_offset(index, &self.dim_offsets) }
    }
}

impl<'a> IndexMut<&[usize]> for VolumeWindowMut<'a> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        unsafe { (*self.block).index_offset_mut(index, &self.dim_offsets) }
    }
}

pub struct Rows<'a, 'b> {
    idx: usize,
    dim: usize,
    stride: usize,
    row_len: usize,
    num_rows: usize,
    row_idx: Vec<usize>,
    dims: &'a [usize],
    dim_offsets: &'a [usize],
    block: *const VolumeBlock,
    _phantom: PhantomData<&'b VolumeBlock>,
}

impl<'a, 'b> Iterator for Rows<'a, 'b> {
    type Item = Row<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.num_rows {
            self.idx += 1;

            let row_start =
                unsafe { (*self.block).flatten_idx_with_offset(&self.row_idx, self.dim_offsets) };

            for (i, idx) in self.row_idx.iter_mut().enumerate() {
                if i != self.dim {
                    if *idx + 1 < self.dims[i] {
                        *idx += 1;
                        break;
                    } else {
                        *idx = 0;
                    }
                }
            }

            Some(Row {
                stride: self.stride,
                row_len: self.row_len,
                row_start,
                block: self.block,
                _phantom: PhantomData,
            })
        } else {
            None
        }
    }
}

pub struct RowsMut<'a, 'b> {
    idx: usize,
    dim: usize,
    stride: usize,
    row_len: usize,
    num_rows: usize,
    row_idx: Vec<usize>,
    dims: &'a [usize],
    dim_offsets: &'a [usize],
    block: *mut VolumeBlock,
    _phantom: PhantomData<&'b mut VolumeBlock>,
}

impl<'a, 'b> Iterator for RowsMut<'a, 'b> {
    type Item = RowMut<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.num_rows {
            self.idx += 1;

            let row_start =
                unsafe { (*self.block).flatten_idx_with_offset(&self.row_idx, self.dim_offsets) };

            for (i, idx) in self.row_idx.iter_mut().enumerate() {
                if i != self.dim {
                    if *idx + 1 < self.dims[i] {
                        *idx += 1;
                        break;
                    } else {
                        *idx = 0;
                    }
                }
            }

            Some(RowMut {
                stride: self.stride,
                row_len: self.row_len,
                row_start,
                block: self.block,
                _phantom: PhantomData,
            })
        } else {
            None
        }
    }
}

pub struct Row<'a> {
    stride: usize,
    row_len: usize,
    row_start: usize,
    block: *const VolumeBlock,
    _phantom: PhantomData<&'a VolumeBlock>,
}

impl<'a> Row<'a> {
    pub fn is_empty(&self) -> bool {
        self.row_len == 0
    }

    pub fn len(&self) -> usize {
        self.row_len
    }

    pub fn as_slice(&self) -> Option<&[f32]> {
        if self.stride == 1 {
            unsafe { Some((*self.block).sub_slice(self.row_start, self.row_len)) }
        } else {
            None
        }
    }

    pub fn iter(&self) -> RowIter<'_, 'a> {
        self.into_iter()
    }
}

impl<'a> Index<usize> for Row<'a> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.row_len);
        let flat_idx = self.row_start + (index * self.stride);
        unsafe { &(*self.block)[flat_idx] }
    }
}

impl<'a, 'b> IntoIterator for &'a Row<'b> {
    type Item = &'a f32;

    type IntoIter = RowIter<'a, 'b>;

    fn into_iter(self) -> Self::IntoIter {
        RowIter { idx: 0, row: self }
    }
}

pub struct RowIter<'a, 'b> {
    idx: usize,
    row: &'a Row<'b>,
}

impl<'a, 'b> Iterator for RowIter<'a, 'b> {
    type Item = &'a f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.row.row_len {
            let idx = self.idx;
            self.idx += 1;

            let flat_idx = self.row.row_start + (idx * self.row.stride);
            let val = unsafe { &(*self.row.block)[flat_idx] };
            Some(val)
        } else {
            None
        }
    }
}

pub struct RowMut<'a> {
    stride: usize,
    row_len: usize,
    row_start: usize,
    block: *mut VolumeBlock,
    _phantom: PhantomData<&'a mut VolumeBlock>,
}

impl<'a> RowMut<'a> {
    pub fn is_empty(&self) -> bool {
        self.row_len == 0
    }

    pub fn len(&self) -> usize {
        self.row_len
    }

    pub fn as_slice(&self) -> Option<&[f32]> {
        if self.stride == 1 {
            unsafe { Some((*self.block).sub_slice(self.row_start, self.row_len)) }
        } else {
            None
        }
    }

    pub fn as_slice_mut(&mut self) -> Option<&mut [f32]> {
        if self.stride == 1 {
            unsafe { Some((*self.block).sub_slice_mut(self.row_start, self.row_len)) }
        } else {
            None
        }
    }

    pub fn iter(&mut self) -> RowIterMut<'_, 'a> {
        self.into_iter()
    }
}

impl<'a> Index<usize> for RowMut<'a> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.row_len);
        let flat_idx = self.row_start + (index * self.stride);
        unsafe { &(*self.block)[flat_idx] }
    }
}

impl<'a> IndexMut<usize> for RowMut<'a> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.row_len);
        let flat_idx = self.row_start + (index * self.stride);
        unsafe { &mut (*self.block)[flat_idx] }
    }
}

impl<'a, 'b> IntoIterator for &'a mut RowMut<'b> {
    type Item = &'a mut f32;

    type IntoIter = RowIterMut<'a, 'b>;

    fn into_iter(self) -> Self::IntoIter {
        RowIterMut { idx: 0, row: self }
    }
}

pub struct RowIterMut<'a, 'b> {
    idx: usize,
    row: &'a mut RowMut<'b>,
}

impl<'a, 'b> Iterator for RowIterMut<'a, 'b> {
    type Item = &'a mut f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.row.row_len {
            let idx = self.idx;
            self.idx += 1;

            let flat_idx = self.row.row_start + (idx * self.row.stride);
            let val = unsafe { &mut (*self.row.block)[flat_idx] };
            Some(val)
        } else {
            None
        }
    }
}
