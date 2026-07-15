use std::collections::HashMap;
use std::fmt;

use crate::error::{Error, Result};
use crate::index::Index;
use crate::series::Series;
use rusanta_core::traits::numeric::Numeric;

/// A DataFrame is a 2-dimensional labeled data structure.
///
/// - Columns are named Series
/// - All Series share the same Index
/// - Column-wise typing (homogeneous per Series)
#[derive(Clone)]
pub struct DataFrame<T>
where
T: Numeric,
{
    columns: HashMap<String, Series<T>>,
    index: Index,
}

impl<T> DataFrame<T>
where
T: Numeric,
{
    /// Create an empty DataFrame.
    pub fn empty() -> Self {
        Self {
            columns: HashMap::new(),
            index: Index::range(0),
        }
    }

    /// Create a DataFrame from columns.
    ///
    /// All Series must share the same index.
    pub fn new(columns: HashMap<String, Series<T>>) -> Result<Self> {
        let mut iter = columns.values();

        let index = match iter.next() {
            Some(s) => s.index().clone(),
            None => Index::range(0),
        };

        for s in iter {
            if s.index() != &index {
                return Err(Error::InvalidValue(
                    "all columns must share the same index".into(),
                ));
            }
        }

        Ok(Self { columns, index })
    }

    /// Create a DataFrame from column vectors.
    pub fn from_vecs(
        data: HashMap<String, Vec<T>>,
        index: Option<Index>,
    ) -> Result<Self> {
        let index = match index {
            Some(idx) => idx,
            None => {
                let len = data
                .values()
                .next()
                .map(|v| v.len())
                .unwrap_or(0);
                Index::range(len)
            }
        };

        let mut columns = HashMap::new();
        for (name, vec) in data {
            let s = Series::new(vec, index.clone(), Some(name.clone()))?;
            columns.insert(name, s);
        }

        Self::new(columns)
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.index.len()
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    /// Get index.
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Get column names.
    pub fn columns(&self) -> Vec<&String> {
        self.columns.keys().collect()
    }

    /// Check if column exists.
    pub fn has_column<S: AsRef<str>>(&self, name: S) -> bool {
        self.columns.contains_key(name.as_ref())
    }

    /// Get a column by name.
    pub fn column<S: AsRef<str>>(&self, name: S) -> Result<&Series<T>> {
        self.columns
        .get(name.as_ref())
        .ok_or_else(|| Error::KeyNotFound(name.as_ref().into()))
    }

    /// Get a mutable column.
    pub fn column_mut<S: AsRef<str>>(&mut self, name: S) -> Result<&mut Series<T>> {
        self.columns
        .get_mut(name.as_ref())
        .ok_or_else(|| Error::KeyNotFound(name.as_ref().into()))
    }

    /// Insert or replace a column.
    pub fn insert<S: Into<String>>(&mut self, name: S, series: Series<T>) -> Result<()> {
        if series.index() != &self.index {
            return Err(Error::InvalidValue(
                "inserted series index does not match dataframe index".into(),
            ));
        }

        let name = name.into();
        self.columns.insert(name.clone(), series);
        Ok(())
    }

    /// Select a subset of columns.
    pub fn select<S: AsRef<str>>(&self, names: &[S]) -> Result<Self> {
        let mut cols = HashMap::new();

        for name in names {
            let s = self.column(name)?;
            cols.insert(name.as_ref().to_string(), s.clone());
        }

        Ok(Self {
            columns: cols,
            index: self.index.clone(),
        })
    }

    /// Select rows by position.
    pub fn take(&self, positions: &[usize]) -> Result<Self> {
        let mut cols = HashMap::new();

        for (name, series) in &self.columns {
            cols.insert(name.clone(), series.take(positions)?);
        }

        Ok(Self::new(cols)?)
    }

    /// Align two DataFrames by index (inner join semantics).
    pub fn align(&self, other: &Self) -> Result<(Self, Self)> {
        let (l_idx, r_idx) = self.index.align(&other.index);

        Ok((self.take(&l_idx)?, other.take(&r_idx)?))
    }

    /// Apply a function column-wise.
    pub fn map<F>(&self, f: F) -> Result<Self>
    where
    F: Fn(&Series<T>) -> Result<Series<T>>,
    {
        let mut cols = HashMap::new();

        for (name, series) in &self.columns {
            let mut out = f(series)?;
            out.set_name(name.clone());
            cols.insert(name.clone(), out);
        }

        Ok(Self::new(cols)?)
    }

    /// Convert DataFrame to row-major vectors.
    pub fn to_rows(&self) -> Result<Vec<Vec<T>>> {
        let mut rows = vec![Vec::with_capacity(self.ncols()); self.nrows()];

        for col in self.columns.values() {
            for i in 0..self.nrows() {
                rows[i].push(col.get(i)?);
            }
        }

        Ok(rows)
    }
}

impl<T> fmt::Debug for DataFrame<T>
where
T: Numeric,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DataFrame")
        .field("nrows", &self.nrows())
        .field("ncols", &self.ncols())
        .field("columns", &self.columns.keys())
        .finish()
    }
}

impl<T> PartialEq for DataFrame<T>
where
T: Numeric + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.columns == other.columns
    }
}
