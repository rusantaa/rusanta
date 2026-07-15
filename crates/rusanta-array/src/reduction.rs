use std::ops::{Add, Sub, Mul, Div};

use rusanta_core::{Numeric, Result, RusantaError};
use crate::ndarray::NdArray;
use crate::broadcast::broadcast;

/// Elementwise binary operation core
fn elementwise_op<T, F>(
    lhs: &NdArray<T>,
    rhs: &NdArray<T>,
    op: F,
) -> Result<NdArray<T>>
where
T: Numeric,
F: Fn(T, T) -> T,
{
    let (a, b) = broadcast(lhs, rhs)?;

    let size = a.size();
    let mut out_buf = Vec::with_capacity(size);

    for i in 0..size {
        let av = a.get(i)?;
        let bv = b.get(i)?;
        out_buf.push(op(av, bv));
    }

    Ok(NdArray::new(
        out_buf.into(),
                    a.shape().to_vec(),
    )?)
}

/// Elementwise addition
pub fn add<T>(lhs: &NdArray<T>, rhs: &NdArray<T>) -> Result<NdArray<T>>
where
T: Numeric + Add<Output = T>,
{
    elementwise_op(lhs, rhs, |a, b| a + b)
}

/// Elementwise subtraction
pub fn sub<T>(lhs: &NdArray<T>, rhs: &NdArray<T>) -> Result<NdArray<T>>
where
T: Numeric + Sub<Output = T>,
{
    elementwise_op(lhs, rhs, |a, b| a - b)
}

/// Elementwise multiplication
pub fn mul<T>(lhs: &NdArray<T>, rhs: &NdArray<T>) -> Result<NdArray<T>>
where
T: Numeric + Mul<Output = T>,
{
    elementwise_op(lhs, rhs, |a, b| a * b)
}

/// Elementwise division
pub fn div<T>(lhs: &NdArray<T>, rhs: &NdArray<T>) -> Result<NdArray<T>>
where
T: Numeric + Div<Output = T>,
{
    elementwise_op(lhs, rhs, |a, b| a / b)
}

/* ===========================
 * Operator overloading
 * =========================== */

impl<T> Add for &NdArray<T>
where
T: Numeric + Add<Output = T>,
{
    type Output = NdArray<T>;

    fn add(self, rhs: Self) -> Self::Output {
        add(self, rhs).expect("NdArray add failed")
    }
}

impl<T> Sub for &NdArray<T>
where
T: Numeric + Sub<Output = T>,
{
    type Output = NdArray<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        sub(self, rhs).expect("NdArray sub failed")
    }
}

impl<T> Mul for &NdArray<T>
where
T: Numeric + Mul<Output = T>,
{
    type Output = NdArray<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        mul(self, rhs).expect("NdArray mul failed")
    }
}

impl<T> Div for &NdArray<T>
where
T: Numeric + Div<Output = T>,
{
    type Output = NdArray<T>;

    fn div(self, rhs: Self) -> Self::Output {
        div(self, rhs).expect("NdArray div failed")
    }
}
