use std::ops::{
    Add, Sub, Mul, Div, Rem,
    AddAssign, SubAssign, MulAssign, DivAssign,
    Neg,
};

/// Base trait for all numeric types in Rusanta.
///
/// This trait is intentionally minimal.
/// It represents values that can live in arrays and datasets.
pub trait Numeric:
Copy
+ Clone
+ Send
+ Sync
+ PartialEq
+ PartialOrd
+ Add<Output = Self>
+ Sub<Output = Self>
+ Mul<Output = Self>
+ Div<Output = Self>
+ AddAssign
+ SubAssign
+ MulAssign
+ DivAssign
{
}

impl<T> Numeric for T where
T: Copy
+ Clone
+ Send
+ Sync
+ PartialEq
+ PartialOrd
+ Add<Output = T>
+ Sub<Output = T>
+ Mul<Output = T>
+ Div<Output = T>
+ AddAssign
+ SubAssign
+ MulAssign
+ DivAssign
{
}

/// Trait for integer-like numeric types.
pub trait Integer:
Numeric
+ Rem<Output = Self>
{
    /// Zero value.
    fn zero() -> Self;

    /// One value.
    fn one() -> Self;
}

/// Trait for floating-point numeric types.
///
/// This is where statistical and ML-heavy functionality lives.
pub trait Float:
Numeric
+ Neg<Output = Self>
{
    /// Zero value.
    fn zero() -> Self;

    /// One value.
    fn one() -> Self;

    /// Smallest representable difference.
    fn epsilon() -> Self;

    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn powf(self, n: Self) -> Self;

    fn is_nan(self) -> bool;
    fn is_finite(self) -> bool;
}

/// Trait for numeric accumulation.
///
/// This allows summing integers into larger types
/// (e.g. i32 -> i64, f32 -> f64).
pub trait Accumulate {
    type Output: Numeric;

    fn to_acc(self) -> Self::Output;
}

//
// ========================
// Implementations
// ========================
//

// -------- Integers --------

macro_rules! impl_integer {
    ($($t:ty),*) => {
        $(
            impl Integer for $t {
                #[inline]
                fn zero() -> Self { 0 }

                #[inline]
                fn one() -> Self { 1 }
            }

            impl Accumulate for $t {
                type Output = i64;

                #[inline]
                fn to_acc(self) -> Self::Output {
                    self as i64
                }
            }
        )*
    };
}

impl_integer!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

// -------- Floats --------

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            #[inline]
            fn zero() -> Self { 0.0 }

            #[inline]
            fn one() -> Self { 1.0 }

            #[inline]
            fn epsilon() -> Self {
                <$t>::EPSILON
            }

            #[inline]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }

            #[inline]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }

            #[inline]
            fn exp(self) -> Self {
                <$t>::exp(self)
            }

            #[inline]
            fn ln(self) -> Self {
                <$t>::ln(self)
            }

            #[inline]
            fn powf(self, n: Self) -> Self {
                <$t>::powf(self, n)
            }

            #[inline]
            fn is_nan(self) -> bool {
                <$t>::is_nan(self)
            }

            #[inline]
            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }
        }

        impl Accumulate for $t {
            type Output = f64;

            #[inline]
            fn to_acc(self) -> Self::Output {
                self as f64
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);
