// rusanta-tensor/src/lib.rs

//! # Rusanta Tensor
//!
//! Core tensor computation engine for the Rusanta ecosystem.
//!
//! Provides:
//!
//! - Multi-dimensional tensors
//! - Automatic differentiation
//! - CPU/GPU execution
//! - Neural network operations
//! - Loss functions
//! - Optimization algorithms
//!
//! Architecture:
//!
//! ```text
//! Tensor
//!   |
//!   +-- Autograd Graph
//!   |
//!   +-- Operations
//!   |
//!   +-- Device Backend
//!          |
//!          +-- CPU
//!          |
//!          +-- GPU (rusanta-triton)
//! ```


#![allow(dead_code)]



// =====================================================
// Internal Modules
// =====================================================


pub mod tensor;


pub mod device;


pub mod autograd;


pub mod ops;


pub mod optim;





// =====================================================
// Public API
// =====================================================


pub use tensor::{
    Tensor,
    Shape,
    DType,
};



pub use device::{
    Device,
};





// Autograd exports

pub use autograd::{
    backward,
    Graph,
    Node,
};





// Operation exports

pub use ops::{
    math,
    nn,
    loss,
};





// Optimizer exports

pub use optim::{
    Optimizer,
};







// =====================================================
// Error Handling
// =====================================================


/// Tensor crate result type.
pub type Result<T> =
    std::result::Result<T, TensorError>;




/// Tensor execution errors.
#[derive(Debug)]
pub enum TensorError {


    /// Invalid tensor dimensions.
    ShapeMismatch {
        expected:String,
        got:String,
    },



    /// Unsupported operation.
    UnsupportedOperation(
        String
    ),



    /// Invalid device operation.
    DeviceError(
        String
    ),



    /// Autograd failure.
    AutogradError(
        String
    ),



    /// Generic internal error.
    Internal(
        String
    ),
}





impl std::fmt::Display for TensorError {


    fn fmt(
        &self,
        f:&mut std::fmt::Formatter<'_>,
    )
        -> std::fmt::Result
    {


        match self {


            TensorError::ShapeMismatch {
                expected,
                got,
            } => {

                write!(
                    f,
                    "Shape mismatch: expected {}, got {}",
                    expected,
                    got
                )

            }



            TensorError::UnsupportedOperation(op)=>{

                write!(
                    f,
                    "Unsupported tensor operation: {}",
                    op
                )

            }



            TensorError::DeviceError(msg)=>{

                write!(
                    f,
                    "Device error: {}",
                    msg
                )

            }



            TensorError::AutogradError(msg)=>{

                write!(
                    f,
                    "Autograd error: {}",
                    msg
                )

            }



            TensorError::Internal(msg)=>{

                write!(
                    f,
                    "Internal tensor error: {}",
                    msg
                )

            }

        }

    }

}




impl std::error::Error for TensorError {}





// =====================================================
// Version Information
// =====================================================


pub const VERSION:&str =
    env!("CARGO_PKG_VERSION");



/// Returns crate version.
pub fn version()
    -> &'static str
{
    VERSION
}
