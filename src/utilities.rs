pub fn flatten_idx(dims: &[usize], index: &[usize]) -> usize {
    assert!(index.len() == dims.len());

    if index
        .iter()
        .zip(dims.iter())
        .any(|(&idx, &dim_len)| idx > dim_len)
    {
        panic!("out of bounds");
    }

    unsafe { flatten_idx_unchecked(dims, index) }
}

pub unsafe fn flatten_idx_unchecked(dims: &[usize], index: &[usize]) -> usize {
    let mut idx = index[0];
    let mut offset_multiplier = 1;
    for (i, &dim_idx) in index.iter().enumerate().skip(1) {
        offset_multiplier *= dims[i - 1];
        idx += offset_multiplier * dim_idx;
    }

    idx
}

pub fn flatten_idx_with_offset(dims: &[usize], index: &[usize], offset: &[usize]) -> usize {
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

    unsafe { flatten_idx_with_offset_unchecked(dims, index, offset) }
}

pub unsafe fn flatten_idx_with_offset_unchecked(
    dims: &[usize],
    index: &[usize],
    offset: &[usize],
) -> usize {
    let mut idx = index[0] + offset[0];
    let mut offset_multiplier = 1;
    for (i, (&dim_idx, &offset)) in index.iter().zip(offset.iter()).enumerate().skip(1) {
        offset_multiplier *= dims[i - 1];
        idx += offset_multiplier * (dim_idx + offset);
    }

    idx
}
