use rusanta_core::{Result, RusantaError};
use crate::ndarray::NdArray;

/// Result of broadcasting two arrays.
#[derive(Debug)]
pub struct BroadcastSpec {
    pub shape: Vec<usize>,
    pub lhs_strides: Vec<isize>,
    pub rhs_strides: Vec<isize>,
}

/// Computes broadcasting rules between two shapes.
///
/// Matches NumPy semantics exactly.
pub fn broadcast_shapes(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    lhs_strides: &[isize],
    rhs_strides: &[isize],
) -> Result<BroadcastSpec> {
    let ndim = lhs_shape.len().max(rhs_shape.len());

    let mut shape = Vec::with_capacity(ndim);
    let mut lhs_new_strides = Vec::with_capacity(ndim);
    let mut rhs_new_strides = Vec::with_capacity(ndim);

    for i in 0..ndim {
        let lhs_i = lhs_shape.len().checked_sub(1 + i);
        let rhs_i = rhs_shape.len().checked_sub(1 + i);

        let lhs_dim = lhs_i.map(|i| lhs_shape[i]).unwrap_or(1);
        let rhs_dim = rhs_i.map(|i| rhs_shape[i]).unwrap_or(1);

        let lhs_stride = lhs_i.map(|i| lhs_strides[i]).unwrap_or(0);
        let rhs_stride = rhs_i.map(|i| rhs_strides[i]).unwrap_or(0);

        if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
            return Err(RusantaError::ShapeMismatch {
                expected: lhs_shape.to_vec(),
                       found: rhs_shape.to_vec(),
            });
        }

        let out_dim = lhs_dim.max(rhs_dim);
        shape.push(out_dim);

        lhs_new_strides.push(if lhs_dim == 1 { 0 } else { lhs_stride });
        rhs_new_strides.push(if rhs_dim == 1 { 0 } else { rhs_stride });
    }

    shape.reverse();
    lhs_new_strides.reverse();
    rhs_new_strides.reverse();

    Ok(BroadcastSpec {
        shape,
       lhs_strides: lhs_new_strides,
       rhs_strides: rhs_new_strides,
    })
}

/// Broadcasts two NdArrays into compatible views.
///
/// This is ZERO-COPY.
pub fn broadcast<'a, T>(
    lhs: &'a NdArray<T>,
    rhs: &'a NdArray<T>,
) -> Result<(NdArray<T>, NdArray<T>)>
where
T: rusanta_core::Numeric,
{
    let spec = broadcast_shapes(
        lhs.shape(),
                                rhs.shape(),
                                lhs.strides(),
                                rhs.strides(),
    )?;

    let lhs_view = NdArray {
        buffer: lhs.buffer.clone(),
        shape: spec.shape.clone(),
        strides: spec.lhs_strides,
        offset: lhs.offset,
    };

    let rhs_view = NdArray {
        buffer: rhs.buffer.clone(),
        shape: spec.shape,
        strides: spec.rhs_strides,
        offset: rhs.offset,
    };

    Ok((lhs_view, rhs_view))
}
