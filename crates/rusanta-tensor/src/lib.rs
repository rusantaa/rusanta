// rusanta-tensor/src/lib.rs

//! Rusanta Tensor Engine.
//!
//! Provides:
//!
//! - Tensor computation
//! - Automatic differentiation
//! - CPU/GPU devices
//! - Neural network operations
//! - Optimization algorithms


pub mod tensor;


pub mod device;


pub mod autograd;


pub mod ops;


pub mod optim;







// =====================================================
// Public Tensor API
// =====================================================



pub use tensor::{
    Tensor,
    Shape,
    DType,
    Storage,
};







// =====================================================
// Device API
// =====================================================



pub use device::Device;







// =====================================================
// Autograd API
// =====================================================



pub use autograd::{
    backward,
    Node,
};







// =====================================================
// Optimization API
// =====================================================



pub use optim::{
    Optimizer,
    Parameter,
};


pub use optim::{
    SGD,
    MomentumSGD,
    Adam,
    AdamW,
    RMSProp,
    AdaGrad,
};







// =====================================================
// Operation API
// =====================================================



pub use ops::{
    math,
    nn,
    loss,
};
