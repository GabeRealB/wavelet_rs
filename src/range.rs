use std::{
    iter::Product,
    ops::{AddAssign, Range, Sub},
};

use num_traits::One;

pub fn for_each_range<T>(range: impl Iterator<Item = Range<T>> + Clone, mut f: impl FnMut(&[T]))
where
    T: One + AddAssign + Sub<Output = T> + PartialOrd + Clone,
    usize: Product<T>,
{
    for_each_range_enumerate(range, move |_, e| f(e))
}

pub fn for_each_range_enumerate<T>(
    range: impl Iterator<Item = Range<T>> + Clone,
    mut f: impl FnMut(usize, &[T]),
) where
    T: One + AddAssign + Sub<Output = T> + PartialOrd + Clone,
    usize: Product<T>,
{
    let mut idx: Vec<_> = range.clone().map(|r| r.start).collect();
    let num_elements = range.clone().map(|r| r.end - r.start).product();

    for i in 0..num_elements {
        f(i, &idx);

        for (idx, range) in idx.iter_mut().zip(range.clone()) {
            *idx += T::one();
            if *idx == range.end {
                *idx = range.start.clone();
            }

            if *idx != range.start {
                break;
            }
        }
    }
}

pub fn for_each_range_par<T>(
    range: impl Iterator<Item = Range<T>> + Send + Clone,
    f: impl FnOnce(&[T]) + Send + Clone,
) where
    T: One + AddAssign + Sub<Output = T> + PartialOrd + Clone + Send,
    usize: Product<T>,
{
    for_each_range_par_enumerate(range, move |_, e| f(e))
}

pub fn for_each_range_par_enumerate<T>(
    range: impl Iterator<Item = Range<T>> + Send + Clone,
    f: impl FnOnce(usize, &[T]) + Send + Clone,
) where
    T: One + AddAssign + Sub<Output = T> + PartialOrd + Clone + Send,
    usize: Product<T>,
{
    rayon::scope(move |s| {
        let mut idx: Vec<_> = range.clone().map(|r| r.start).collect();
        let num_elements = range.clone().map(|r| r.end - r.start).product();

        for i in 0..num_elements {
            {
                let f = f.clone();
                let idx = idx.clone();
                s.spawn(move |_| f(i, &idx));
            }

            for (idx, range) in idx.iter_mut().zip(range.clone()) {
                *idx += T::one();
                if *idx == range.end {
                    *idx = range.start.clone();
                }

                if *idx != range.start {
                    break;
                }
            }
        }
    })
}
