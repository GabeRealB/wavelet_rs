use std::{
    iter::Sum,
    ops::{Add, Mul},
};

pub fn strides_for_dims(dims: &[usize]) -> Vec<usize> {
    std::iter::once(1)
        .chain(dims.iter().scan(1usize, |s, &d| {
            *s *= d;
            Some(*s)
        }))
        .take(dims.len())
        .collect()
}

pub fn flatten_idx(dims: &[usize], strides: &[usize], index: &[usize]) -> usize {
    assert_eq!(index.len(), dims.len());
    assert_eq!(strides.len(), dims.len());

    if index
        .iter()
        .zip(dims.iter())
        .any(|(&idx, &dim_len)| idx > dim_len)
    {
        panic!("out of bounds");
    }

    unsafe { flatten_idx_unchecked(strides, index) }
}

pub unsafe fn flatten_idx_unchecked(strides: &[usize], index: &[usize]) -> usize {
    flatten_idx_impl(
        strides.iter().copied(),
        index.iter().copied(),
        std::iter::repeat(0),
    )
}

pub fn flatten_idx_with_offset(
    dims: &[usize],
    strides: &[usize],
    index: &[usize],
    offset: &[usize],
) -> usize {
    assert!(offset.len() == dims.len());
    assert!(index.len() == dims.len());

    if index
        .iter()
        .zip(offset.iter())
        .zip(dims.iter())
        .any(|((&idx, &offset), &dim_len)| (idx + offset) > dim_len)
    {
        panic!("out of bounds");
    }

    unsafe { flatten_idx_with_offset_unchecked(strides, index, offset) }
}

pub unsafe fn flatten_idx_with_offset_unchecked(
    strides: &[usize],
    index: &[usize],
    offset: &[usize],
) -> usize {
    flatten_idx_impl(
        strides.iter().copied(),
        index.iter().copied(),
        offset.iter().copied(),
    )
}

fn flatten_idx_impl<T, U, V>(
    s: impl Iterator<Item = T>,
    i: impl Iterator<Item = U>,
    o: impl Iterator<Item = V>,
) -> T::Output
where
    T::Output: Sum,
    T: Mul<U::Output>,
    U: Add<V>,
{
    s.zip(i).zip(o).map(|((s, i), o)| s * (i + o)).sum()
}

pub fn next_multiple_of(x: usize, y: usize) -> usize {
    let remainder = x % y;
    if remainder == 0 {
        x
    } else {
        x + y - remainder
    }
}
