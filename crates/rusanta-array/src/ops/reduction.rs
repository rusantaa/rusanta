use rusanta_core::{Numeric, Result, RusantaError};
use crate::ndarray::NdArray;

/// Matrix multiplication (2D only).
///
/// Semantics:
/// - (m × k) @ (k × n) → (m × n)
/// - row-major default
/// - supports non-contiguous arrays via strides
pub fn matmul<T>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>>
where
T: Numeric,
{
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(RusantaError::InvalidValue {
            message: "matmul requires 2D arrays".into(),
        });
    }

    let (m, k1) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = (b.shape()[0], b.shape()[1]);

    if k1 != k2 {
        return Err(RusantaError::DimensionMismatch {
            lhs: k1,
            rhs: k2,
        });
    }

    let mut out = NdArray::zeros(&[m, n])?;

    for i in 0..m {
        for j in 0..n {
            let mut acc = T::zero();
            for k in 0..k1 {
                let av = a.get(&[i, k])?;
                let bv = b.get(&[k, j])?;
                acc = acc + av * bv;
            }
            out.set(&[i, j], acc)?;
        }
    }

    Ok(out)
}

/// Dot product of two 1D arrays.
///
/// Semantics:
/// - (n,) · (n,) → scalar
pub fn dot<T>(a: &NdArray<T>, b: &NdArray<T>) -> Result<T>
where
T: Numeric,
{
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(RusantaError::InvalidValue {
            message: "dot requires 1D arrays".into(),
        });
    }

    if a.shape()[0] != b.shape()[0] {
        return Err(RusantaError::DimensionMismatch {
            lhs: a.shape()[0],
                   rhs: b.shape()[0],
        });
    }

    let mut acc = T::zero();
    for i in 0..a.shape()[0] {
        acc = acc + a.get(&[i])? * b.get(&[i])?;
    }

    Ok(acc)
}

/// Transpose a 2D array (view, zero-copy).
///
/// (m × n) → (n × m)
pub fn transpose<T>(a: &NdArray<T>) -> Result<NdArray<T>>
where
T: Numeric,
{
    if a.ndim() != 2 {
        return Err(RusantaError::InvalidValue {
            message: "transpose requires 2D array".into(),
        });
    }

    let shape = vec![a.shape()[1], a.shape()[0]];
    let strides = vec![a.strides()[1], a.strides()[0]];

    Ok(NdArray {
        buffer: a.buffer.clone(),
       shape,
       strides,
       offset: a.offset,
    })
}
