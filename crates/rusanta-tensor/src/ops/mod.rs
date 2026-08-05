// rusanta-tensor/src/ops/mod.rs

//! Tensor operation modules.
//!
//! Operations are separated into:
//!
//! - math
//!     Basic tensor mathematics
//!
//! - nn
//!     Neural network operations
//!
//! - loss
//!     Training objective functions
//!
//!
//! Every operation can optionally create
//! an autograd graph node.

pub mod math;

pub mod nn;

pub mod loss;





// =====================================================
// Common Operation Traits
// =====================================================


use crate::{
    Tensor,
    Result,
};





/// Trait implemented by tensor operations.
///
/// Allows future extension:
///
/// - custom operators
/// - plugin kernels
/// - GPU kernels
pub trait Operation {


    /// Execute operation.
    fn forward(
        &self,
        inputs:&[Tensor],
    )
        -> Result<Tensor>;





    /// Operation name.
    fn name(
        &self,
    )
        -> &'static str;

}





// =====================================================
// Utility Functions
// =====================================================



/// Ensure tensors are compatible.
pub(crate) fn check_same_shape(
    a:&Tensor,
    b:&Tensor,
)
{

    assert_eq!(
        a.shape(),
        b.shape(),
        "Tensor shape mismatch"
    );

}






/// Ensure floating point tensors.
pub(crate) fn require_float(
    tensor:&Tensor,
)
{

    tensor.assert_float();

}
