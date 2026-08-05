// rusanta-tensor/src/tensor.rs

//! Core Tensor implementation for Rusanta.
//!
//! Tensor is the fundamental data structure for:
//! - numerical computation
//! - automatic differentiation
//! - neural networks
//! - GPU execution
//!
//! Design inspired by:
//! - PyTorch Tensor
//! - NumPy ndarray
//! - JAX DeviceArray

use std::fmt;
use std::sync::{Arc, Mutex};

use crate::device::Device;

use crate::autograd::node::Node;

/// Tensor data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float32,
    Float64,
    Int32,
    Int64,
}

/// Raw tensor storage.
///
/// Future versions can replace Vec storage with:
/// - CPU allocator
/// - CUDA memory
/// - WGPU buffer
/// - unified memory
#[derive(Debug, Clone)]
pub enum Storage {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}


/// Tensor shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}


impl Shape {

    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }


    pub fn from_slice(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }


    pub fn ndim(&self) -> usize {
        self.dims.len()
    }


    pub fn dims(&self) -> &[usize] {
        &self.dims
    }


    pub fn numel(&self) -> usize {
        self.dims
            .iter()
            .product()
    }
}



/// Tensor object.
///
/// Contains:
/// - data
/// - shape
/// - dtype
/// - device
/// - gradient
/// - autograd graph reference
pub struct Tensor {

    storage: Storage,

    shape: Shape,

    dtype: DType,

    device: Device,


    requires_grad: bool,


    grad: Option<Box<Tensor>>,


    /// Connection to computation graph.
    ///
    /// None means leaf tensor.
    node: Option<Arc<Mutex<Node>>>,
}



impl Tensor {


    // -------------------------------------------------
    // Constructors
    // -------------------------------------------------


    pub fn zeros(
        shape: &[usize],
        dtype: DType,
        device: Device,
    ) -> Self {

        let size =
            shape.iter().product();


        let storage =
            match dtype {

                DType::Float32 =>
                    Storage::F32(
                        vec![0.0; size]
                    ),

                DType::Float64 =>
                    Storage::F64(
                        vec![0.0; size]
                    ),

                DType::Int32 =>
                    Storage::I32(
                        vec![0; size]
                    ),

                DType::Int64 =>
                    Storage::I64(
                        vec![0; size]
                    ),
            };


        Self {

            storage,

            shape:
                Shape::from_slice(shape),

            dtype,

            device,

            requires_grad:false,

            grad:None,

            node:None,
        }
    }




    pub fn ones(
        shape:&[usize],
        dtype:DType,
        device:Device,
    ) -> Self {


        let size =
            shape.iter().product();


        let storage =
            match dtype {

                DType::Float32 =>
                    Storage::F32(
                        vec![1.0;size]
                    ),

                DType::Float64 =>
                    Storage::F64(
                        vec![1.0;size]
                    ),

                DType::Int32 =>
                    Storage::I32(
                        vec![1;size]
                    ),

                DType::Int64 =>
                    Storage::I64(
                        vec![1;size]
                    ),
            };


        Self {

            storage,

            shape:
                Shape::from_slice(shape),

            dtype,

            device,

            requires_grad:false,

            grad:None,

            node:None,
        }
    }




    pub fn from_vec_f32(
        data:Vec<f32>,
        shape:&[usize],
        device:Device,
    )->Self {


        assert_eq!(
            data.len(),
            shape.iter().product()
        );


        Self {

            storage:
                Storage::F32(data),

            shape:
                Shape::from_slice(shape),

            dtype:
                DType::Float32,

            device,

            requires_grad:false,

            grad:None,

            node:None,
        }
    }



    pub fn from_vec_f64(
        data:Vec<f64>,
        shape:&[usize],
        device:Device,
    )->Self {


        assert_eq!(
            data.len(),
            shape.iter().product()
        );


        Self {

            storage:
                Storage::F64(data),

            shape:
                Shape::from_slice(shape),

            dtype:
                DType::Float64,

            device,

            requires_grad:false,

            grad:None,

            node:None,
        }
    }




    // -------------------------------------------------
    // Metadata
    // -------------------------------------------------


    pub fn shape(&self)
        -> &Shape
    {
        &self.shape
    }



    pub fn ndim(&self)
        -> usize
    {
        self.shape.ndim()
    }



    pub fn numel(&self)
        -> usize
    {
        self.shape.numel()
    }



    pub fn dtype(&self)
        -> DType
    {
        self.dtype
    }



    pub fn device(&self)
        -> Device
    {
        self.device
    }



    pub fn requires_grad(&self)
        -> bool
    {
        self.requires_grad
    }



    pub fn set_requires_grad(
        &mut self,
        value:bool,
    )
    {
        self.requires_grad=value;
    }

}
