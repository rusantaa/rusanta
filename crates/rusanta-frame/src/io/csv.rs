use crate::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::series::Series;
use rusanta_core::traits::numeric::Numeric;

/// Statistical operations for Series.
impl<T> Series<T>
where
T: Numeric,
{
    /// Sum of values.
    pub fn sum(&self) -> Result<T> {
        let mut acc = T::zero();
        for i in 0..self.len() {
            acc = acc + self.get(i)?;
        }
        Ok(acc)
    }

    /// Mean (average) of values.
    pub fn mean(&self) -> Result<T> {
        if self.len() == 0 {
            return Err(Error::InvalidValue("cannot compute mean of empty series".into()));
        }
        Ok(self.sum()? / T::from_usize(self.len()))
    }

    /// Minimum value.
    pub fn min(&self) -> Result<T> {
        if self.len() == 0 {
            return Err(Error::InvalidValue("cannot compute min of empty series".into()));
        }

        let mut min = self.get(0)?;
        for i in 1..self.len() {
            let v = self.get(i)?;
            if v < min {
                min = v;
            }
        }
        Ok(min)
    }

    /// Maximum value.
    pub fn max(&self) -> Result<T> {
        if self.len() == 0 {
            return Err(Error::InvalidValue("cannot compute max of empty series".into()));
        }

        let mut max = self.get(0)?;
        for i in 1..self.len() {
            let v = self.get(i)?;
            if v > max {
                max = v;
            }
        }
        Ok(max)
    }

    /// Variance (population variance).
    pub fn var(&self) -> Result<T> {
        let mean = self.mean()?;
        let mut acc = T::zero();

        for i in 0..self.len() {
            let diff = self.get(i)? - mean;
            acc = acc + diff * diff;
        }

        Ok(acc / T::from_usize(self.len()))
    }

    /// Standard deviation.
    pub fn std(&self) -> Result<T> {
        Ok(self.var()?.sqrt())
    }
}

/// Statistical operations for DataFrame.
impl<T> DataFrame<T>
where
T: Numeric,
{
    /// Column-wise sum.
    pub fn sum(&self) -> Result<DataFrame<T>> {
        self.map(|s| {
            let value = s.sum()?;
            Ok(Series::from_vec(vec![value]))
        })
    }

    /// Column-wise mean.
    pub fn mean(&self) -> Result<DataFrame<T>> {
        self.map(|s| {
            let value = s.mean()?;
            Ok(Series::from_vec(vec![value]))
        })
    }

    /// Column-wise min.
    pub fn min(&self) -> Result<DataFrame<T>> {
        self.map(|s| {
            let value = s.min()?;
            Ok(Series::from_vec(vec![value]))
        })
    }

    /// Column-wise max.
    pub fn max(&self) -> Result<DataFrame<T>> {
        self.map(|s| {
            let value = s.max()?;
            Ok(Series::from_vec(vec![value]))
        })
    }

    /// Column-wise variance.
    pub fn var(&self) -> Result<DataFrame<T>> {
        self.map(|s| {
            let value = s.var()?;
            Ok(Series::from_vec(vec![value]))
        })
    }

    /// Column-wise standard deviation.
    pub fn std(&self) -> Result<DataFrame<T>> {
        self.map(|s| {
            let value = s.std()?;
            Ok(Series::from_vec(vec![value]))
        })
    }
}
