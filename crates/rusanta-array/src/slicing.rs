use rusanta_core::{Result, RusantaError};
use crate::ndarray::NdArray;

/// Slice specification for NdArray views (NumPy-like).
#[derive(Debug, Clone)]
pub enum Slice {
    /// Select a single index, removing the axis.
    Index(usize),

    /// Select a range with optional stride.
    Range {
        start: usize,
        end: usize,
        step: usize,
    },

    /// Select the full axis.
    All,
}

/// Apply slicing to an NdArray, producing a zero-copy view.
///
/// This function:
/// - validates slices
/// - computes new shape
/// - computes new strides
/// - computes new offset
///
/// NumPy-compatible semantics.
pub fn apply_slice<T>(
    arr: &NdArray<T>,
    slices: &[Slice],
) -> Result<NdArray<T>>
where
T: rusanta_core::Numeric,
{
    if slices.len() > arr.ndim() {
        return Err(RusantaError::DimensionMismatch {
            lhs: slices.len(),
                   rhs: arr.ndim(),
        });
    }

    let mut new_shape = Vec::new();
    let mut new_strides = Vec::new();
    let mut new_offset = arr.offset as isize;

    for (axis, slice) in slices.iter().enumerate() {
        let dim = arr.shape()[axis];
        let stride = arr.strides()[axis];

        match *slice {
            Slice::Index(i) => {
                if i >= dim {
                    return Err(RusantaError::IndexOutOfBounds {
                        index: i,
                        len: dim,
                    });
                }
                new_offset += i as isize * stride;
            }

            Slice::Range { start, end, step } => {
                if step == 0 {
                    return Err(RusantaError::InvalidValue {
                        message: "slice step must be non-zero".into(),
                    });
                }
                if start >= end || end > dim {
                    return Err(RusantaError::InvalidValue {
                        message: "invalid slice range".into(),
                    });
                }

                let len = (end - start + step - 1) / step;
                new_shape.push(len);
                new_strides.push(stride * step as isize);
                new_offset += start as isize * stride;
            }

            Slice::All => {
                new_shape.push(dim);
                new_strides.push(stride);
            }
        }
    }

    // Remaining axes (implicit Slice::All)
    for axis in slices.len()..arr.ndim() {
        new_shape.push(arr.shape()[axis]);
        new_strides.push(arr.strides()[axis]);
    }

    Ok(NdArray {
        buffer: arr.buffer.clone(),
       shape: new_shape,
       strides: new_strides,
       offset: new_offset as usize,
    })
}
