use std::fmt;

use crate::error::{Error, Result};
use crate::index::Index;
use rusanta_array::ndarray::NdArray;
use rusanta_core::traits::numeric::Numeric;

/// A Series is a 1-dimensional labeled array.
///
/// Conceptually equivalent to `pandas.Series`.
/// - Owns data
/// - Owns an Index
/// - Strongly typed
#[derive(Clone)]
pub struct Series<T>
where
T: Numeric,
{
    name: Option<String>,
    data: NdArray<T>,
    index: Index,
}

impl<T> Series<T>
where
T: Numeric,
{
    /// Create a new Series from raw values and index.
    pub fn new(
        data: Vec<T>,
        index: Index,
        name: Option<String>,
    ) -> Result<Self> {
        if data.len() != index.len() {
            return Err(Error::InvalidValue(
                "data length does not match index length".into(),
            ));
        }

        let array = NdArray::from_vec(data, &[index.len()])?;

        Ok(Self {
            name,
            data: array,
            index,
        })
    }

    /// Create a Series with a default integer index.
    pub fn from_vec(data: Vec<T>) -> Self {
        let index = Index::range(data.len());
        let array = NdArray::from_vec(data, &[index.len()]).unwrap();

        Self {
            name: None,
            data: array,
            index,
        }
    }

    /// Length of the series.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Get series name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set series name.
    pub fn set_name<S: Into<String>>(&mut self, name: S) {
        self.name = Some(name.into());
    }

    /// Get index.
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Get underlying data.
    pub fn values(&self) -> &NdArray<T> {
        &self.data
    }

    /// Get value by position.
    pub fn get(&self, pos: usize) -> Result<T> {
        Ok(self.data.get(&[pos])?)
    }

    /// Get value by label.
    pub fn get_label<S: AsRef<str>>(&self, label: S) -> Result<T> {
        let pos = self.index.position(label)?;
        self.get(pos)
    }

    /// Select a subset of the Series by positions.
    pub fn take(&self, positions: &[usize]) -> Result<Self> {
        let mut data = Vec::with_capacity(positions.len());
        let mut labels = Vec::with_capacity(positions.len());

        for &p in positions {
            data.push(self.get(p)?);
            labels.push(self.index.label(p)?.to_string());
        }

        Ok(Series::new(
            data,
            Index::new(labels)?,
                       self.name.clone(),
        )?)
    }

    /// Align two Series by index (inner join semantics).
    pub fn align(&self, other: &Series<T>) -> Result<(Self, Self)> {
        let (l_idx, r_idx) = self.index.align(&other.index);

        let left = self.take(&l_idx)?;
        let right = other.take(&r_idx)?;

        Ok((left, right))
    }

    /// Apply a function element-wise.
    pub fn map<F>(&self, f: F) -> Result<Self>
    where
    F: Fn(T) -> T,
    {
        let mut out = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            out.push(f(self.get(i)?));
        }

        Ok(Series::new(
            out,
            self.index.clone(),
                       self.name.clone(),
        )?)
    }

    /// Convert Series into raw vector (dropping index).
    pub fn to_vec(&self) -> Vec<T> {
        (0..self.len())
        .map(|i| self.get(i).unwrap())
        .collect()
    }
}

impl<T> fmt::Debug for Series<T>
where
T: Numeric + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Series")
        .field("name", &self.name)
        .field("len", &self.len())
        .finish()
    }
}

impl<T> PartialEq for Series<T>
where
T: Numeric + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.to_vec() == other.to_vec()
    }
}
