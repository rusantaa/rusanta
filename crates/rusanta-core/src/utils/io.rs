use crate::error::{Result, RusantaError};
use crate::traits::dataset::Dataset;
use crate::traits::numeric::{Accumulate, Float, Numeric};

/// Computes the sum of a dataset.
pub fn sum<T, D>(data: &D) -> Result<T::Output>
where
T: Numeric + Accumulate,
D: Dataset<T>,
{
    if data.is_empty() {
        return Err(RusantaError::EmptyData {
            context: "sum",
        });
    }

    let mut acc = T::to_acc(data.get(0)?);
    for i in 1..data.len() {
        acc += data.get(i)?.to_acc();
    }
    Ok(acc)
}

/// Computes the mean of a dataset.
pub fn mean<T, D>(data: &D) -> Result<T::Output>
where
T: Numeric + Accumulate,
T::Output: Float,
D: Dataset<T>,
{
    let n = data.len();
    if n == 0 {
        return Err(RusantaError::EmptyData {
            context: "mean",
        });
    }

    let total = sum::<T, D>(data)?;
    Ok(total / T::Output::from(n as f64))
}

/// Computes the variance of a dataset.
pub fn variance<T, D>(data: &D) -> Result<T::Output>
where
T: Numeric + Accumulate,
T::Output: Float,
D: Dataset<T>,
{
    let n = data.len();
    if n < 2 {
        return Err(RusantaError::InvalidValue {
            message: "variance requires at least two elements".into(),
        });
    }

    let mean_val = mean::<T, D>(data)?;

    let mut acc = T::Output::zero();
    for i in 0..n {
        let x = data.get(i)?.to_acc();
        let diff = x - mean_val;
        acc += diff * diff;
    }

    Ok(acc / T::Output::from((n - 1) as f64))
}

/// Computes the minimum value.
pub fn min<T, D>(data: &D) -> Result<T>
where
T: Numeric,
D: Dataset<T>,
{
    if data.is_empty() {
        return Err(RusantaError::EmptyData {
            context: "min",
        });
    }

    let mut m = data.get(0)?;
    for i in 1..data.len() {
        let v = data.get(i)?;
        if v < m {
            m = v;
        }
    }
    Ok(m)
}

/// Computes the maximum value.
pub fn max<T, D>(data: &D) -> Result<T>
where
T: Numeric,
D: Dataset<T>,
{
    if data.is_empty() {
        return Err(RusantaError::EmptyData {
            context: "max",
        });
    }

    let mut m = data.get(0)?;
    for i in 1..data.len() {
        let v = data.get(i)?;
        if v > m {
            m = v;
        }
    }
    Ok(m)
}
