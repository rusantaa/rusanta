use rusanta_core::{
    Buffer,
    Dataset,
    Numeric,
    Result,
    RusantaError,
    View,
};

/// Slice specification for NdArray views (NumPy-like).
#[derive(Debug, Clone)]
pub enum Slice {
    Index(usize),
    Range {
        start: usize,
        end: usize,
        step: usize,
    },
    All,
}

/// A contiguous or strided n-dimensional array.
///
/// This is the core tensor structure of Rusanta.
/// Zero-copy slicing and reshaping are supported.
#[derive(Debug, Clone)]
pub struct NdArray<T: Numeric> {
    buffer: Buffer<T>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}

impl<T: Numeric> NdArray<T> {
    /* ===========================
     * Constructors
     * =========================== */

    /// Creates a new NdArray from a buffer and shape (row-major).
    pub fn new(buffer: Buffer<T>, shape: Vec<usize>) -> Result<Self> {
        let expected: usize = shape.iter().product();
        if buffer.len() != expected {
            return Err(RusantaError::ShapeMismatch {
                expected: vec![expected],
                found: vec![buffer.len()],
            });
        }

        Ok(Self {
            buffer,
            strides: compute_strides(&shape),
           shape,
           offset: 0,
        })
    }

    /// Creates a zero-filled array.
    pub fn zeros(shape: &[usize]) -> Result<Self>
    where
    T: rusanta_core::Float,
    {
        let size: usize = shape.iter().product();
        Ok(Self {
            buffer: Buffer::filled(size, T::zero()),
           shape: shape.to_vec(),
           strides: compute_strides(shape),
           offset: 0,
        })
    }

    /* ===========================
     * Shape & metadata
     * =========================== */

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.strides == compute_strides(&self.shape)
    }

    /* ===========================
     * Indexing
     * =========================== */

    fn linear_index(&self, index: &[usize]) -> Result<usize> {
        if index.len() != self.ndim() {
            return Err(RusantaError::DimensionMismatch {
                lhs: index.len(),
                       rhs: self.ndim(),
            });
        }

        let mut lin = self.offset as isize;

        for ((&i, &dim), &stride) in index
            .iter()
            .zip(self.shape.iter())
            .zip(self.strides.iter())
            {
                if i >= dim {
                    return Err(RusantaError::IndexOutOfBounds {
                        index: i,
                        len: dim,
                    });
                }
                lin += i as isize * stride;
            }

            Ok(lin as usize)
    }

    pub fn get_nd(&self, index: &[usize]) -> Result<T> {
        let lin = self.linear_index(index)?;
        Ok(self.buffer[lin])
    }

    /* ===========================
     * Views
     * =========================== */

    /// Returns a flattened view of the array.
    pub fn flat_view(&self) -> View<'_, T> {
        View::new(
            &self.buffer,
            self.offset,
            self.size(),
                  1,
        )
        .expect("NdArray invariant violated")
    }

    /// Reshapes the array without copying.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.size() {
            return Err(RusantaError::ShapeMismatch {
                expected: vec![self.size()],
                       found: vec![new_size],
            });
        }

        Ok(Self {
            buffer: self.buffer.clone(),
           shape: new_shape.to_vec(),
           strides: compute_strides(new_shape),
           offset: self.offset,
        })
    }

    /* ===========================
     * Slicing
     * =========================== */

    /// Creates a zero-copy sliced view.
    pub fn slice(&self, slices: &[Slice]) -> Result<Self> {
        if slices.len() > self.ndim() {
            return Err(RusantaError::DimensionMismatch {
                lhs: slices.len(),
                       rhs: self.ndim(),
            });
        }

        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();
        let mut new_offset = self.offset as isize;

        for (axis, slice) in slices.iter().enumerate() {
            let dim = self.shape[axis];
            let stride = self.strides[axis];

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
                    if step == 0 || start >= end || end > dim {
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

        for axis in slices.len()..self.ndim() {
            new_shape.push(self.shape[axis]);
            new_strides.push(self.strides[axis]);
        }

        Ok(Self {
            buffer: self.buffer.clone(),
           shape: new_shape,
           strides: new_strides,
           offset: new_offset as usize,
        })
    }

    /* ===========================
     * Axis permutation
     * =========================== */

    /// Permutes axes without copying (transpose is a special case).
    pub fn permute_axes(&self, axes: &[usize]) -> Result<Self> {
        if axes.len() != self.ndim() {
            return Err(RusantaError::DimensionMismatch {
                lhs: axes.len(),
                       rhs: self.ndim(),
            });
        }

        let mut seen = vec![false; self.ndim()];
        for &a in axes {
            if a >= self.ndim() || seen[a] {
                return Err(RusantaError::InvalidValue {
                    message: "invalid axis permutation".into(),
                });
            }
            seen[a] = true;
        }

        let new_shape = axes.iter().map(|&i| self.shape[i]).collect();
        let new_strides = axes.iter().map(|&i| self.strides[i]).collect();

        Ok(Self {
            buffer: self.buffer.clone(),
           shape: new_shape,
           strides: new_strides,
           offset: self.offset,
        })
    }

    /// Transposes the array (reverses axes).
    pub fn transpose(&self) -> Self {
        let axes: Vec<usize> = (0..self.ndim()).rev().collect();
        self.permute_axes(&axes)
        .expect("transpose invariant violated")
    }
}

/* ===========================
 * Dataset implementation
 * =========================== */

impl<T: Numeric> Dataset<T> for NdArray<T> {
    fn len(&self) -> usize {
        self.size()
    }

    fn get(&self, index: usize) -> Result<T> {
        Ok(self.buffer[self.offset + index])
    }
}

/* ===========================
 * Helpers
 * =========================== */

fn compute_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0; shape.len()];
    let mut acc = 1;

    for (i, dim) in shape.iter().enumerate().rev() {
        strides[i] = acc as isize;
        acc *= *dim;
    }

    strides
}
