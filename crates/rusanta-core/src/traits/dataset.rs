use crate::error::Result;
use crate::traits::numeric::Numeric;

/// A read-only abstraction over a collection of numeric data.
///
/// This trait is the foundation for:
/// - arrays
/// - tensors
/// - series / columns
/// - views
///
/// Algorithms should depend on `Dataset`, not concrete containers.
pub trait Dataset<T: Numeric>: Send + Sync {
    /// Returns the total number of elements.
    fn len(&self) -> usize;

    /// Returns true if the dataset is empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the element at index `i`.
    fn get(&self, index: usize) -> Result<T>;

    /// Returns an iterator over the dataset.
    ///
    /// Default implementation provided.
    fn iter(&self) -> DatasetIter<'_, T, Self>
    where
    Self: Sized,
    {
        DatasetIter {
            dataset: self,
            index: 0,
            len: self.len(),
        }
    }
}

/// Iterator over a Dataset.
pub struct DatasetIter<'a, T: Numeric, D: Dataset<T>> {
    dataset: &'a D,
    index: usize,
    len: usize,
}

impl<'a, T: Numeric, D: Dataset<T>> Iterator for DatasetIter<'a, T, D> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            return None;
        }

        let i = self.index;
        self.index += 1;

        self.dataset.get(i).ok()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T: Numeric, D: Dataset<T>> ExactSizeIterator for DatasetIter<'a, T, D> {}
