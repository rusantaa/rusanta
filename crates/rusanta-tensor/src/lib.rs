//! Rusanta Tensor Engine.
//!
//! Core tensor computation library.
//!
//! Features:
//!
//! - Multi-dimensional tensors
//! - Automatic differentiation
//! - CPU/GPU abstraction
//! - Neural network operations
//! - Optimization algorithms


pub mod tensor;

pub mod device;

pub mod autograd;

pub mod ops;

pub mod optim;







// =====================================================
// Tensor exports
// =====================================================


pub use tensor::{
    Tensor,
    Shape,
    DType,
    Storage,
};







// =====================================================
// Device exports
// =====================================================


pub use device::Device;







// =====================================================
// Optimizer exports
// =====================================================


pub use optim::{
    Optimizer,
    Parameter,
};





pub use optim::sgd::SGD;

pub use optim::momentum::MomentumSGD;

pub use optim::adam::Adam;

pub use optim::adamw::AdamW;

pub use optim::rmsprop::RMSProp;

pub use optim::adagrad::AdaGrad;







// =====================================================
// Operation namespaces
// =====================================================


pub use ops::{
    math,
    nn,
    loss,
};
