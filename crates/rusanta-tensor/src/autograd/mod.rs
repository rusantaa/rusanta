// rusanta-tensor/src/autograd/mod.rs

//! Automatic differentiation engine.
//!
//! Provides dynamic computation graphs similar to:
//!
//! - PyTorch Autograd
//! - TensorFlow GradientTape
//! - JAX tracing
//!
//! Flow:
//!
//! ```text
//! Forward pass
//!
//! x -----> op -----> y
//!          |
//!          v
//!        Node
//!
//!
//! Backward pass
//!
//! dy/dy = 1
//!
//! Node backward()
//!          |
//!          v
//!       gradients
//! ```


pub mod node;

pub mod graph;

pub mod backward;





// =====================================================
// Public exports
// =====================================================


pub use node::{
    Node,
    Op,
};


pub use graph::{
    Graph,
};


pub use backward::{
    backward,
};






// =====================================================
// Gradient utilities
// =====================================================


use crate::{
    Tensor,
    Result,
};





/// Enable gradient tracking.
///
/// Equivalent concept:
///
/// PyTorch:
/// ```python
/// tensor.requires_grad_(True)
/// ```
pub fn requires_grad(
    mut tensor:Tensor,
)
    -> Tensor
{

    tensor.set_requires_grad(true);

    tensor

}





/// Execute backward propagation.
///
/// Starting from a scalar output tensor:
///
/// ```text
///
/// loss
///  |
/// backward()
///  |
/// gradients
///
/// ```
pub fn backward_tensor(
    tensor:&Tensor,
)
    -> Result<()>
{

    backward::backward(
        tensor
    )

}
