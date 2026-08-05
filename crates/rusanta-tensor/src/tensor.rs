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

impl Tensor {


    // -------------------------------------------------
    // Storage access
    // -------------------------------------------------


    pub fn storage(&self) -> &Storage {
        &self.storage
    }


    pub fn storage_mut(&mut self) -> &mut Storage {
        &mut self.storage
    }



    pub fn as_f32(&self) -> &[f32] {

        match &self.storage {

            Storage::F32(data) =>
                data.as_slice(),

            _ =>
                panic!(
                    "Tensor dtype is not Float32"
                ),
        }
    }



    pub fn as_f64(&self) -> &[f64] {

        match &self.storage {

            Storage::F64(data) =>
                data.as_slice(),

            _ =>
                panic!(
                    "Tensor dtype is not Float64"
                ),
        }
    }




    pub fn as_f32_mut(&mut self)
        -> &mut [f32]
    {

        match &mut self.storage {

            Storage::F32(data) =>
                data.as_mut_slice(),

            _ =>
                panic!(
                    "Tensor dtype is not Float32"
                ),
        }
    }



    pub fn as_f64_mut(&mut self)
        -> &mut [f64]
    {

        match &mut self.storage {

            Storage::F64(data) =>
                data.as_mut_slice(),

            _ =>
                panic!(
                    "Tensor dtype is not Float64"
                ),
        }
    }




    // -------------------------------------------------
    // Gradient handling
    // -------------------------------------------------


    pub fn grad(
        &self
    )
        -> Option<&Tensor>
    {
        self.grad
            .as_deref()
    }



    pub fn grad_mut(
        &mut self
    )
        -> Option<&mut Tensor>
    {
        self.grad
            .as_deref_mut()
    }




    pub fn zero_grad(
        &mut self
    )
    {

        if let Some(g)=
            self.grad.as_mut()
        {

            match g.storage_mut() {

                Storage::F32(data)=> {
                    for v in data {
                        *v=0.0;
                    }
                }


                Storage::F64(data)=> {
                    for v in data {
                        *v=0.0;
                    }
                }


                Storage::I32(data)=> {
                    for v in data {
                        *v=0;
                    }
                }


                Storage::I64(data)=> {
                    for v in data {
                        *v=0;
                    }
                }

            }
        }
    }





    pub(crate) fn set_grad(
        &mut self,
        grad:Tensor,
    )
    {
        self.grad=
            Some(Box::new(grad));
    }







    // -------------------------------------------------
    // Autograd graph
    // -------------------------------------------------



    pub(crate) fn node(
        &self
    )
        -> Option<Arc<Mutex<Node>>>
    {
        self.node.clone()
    }



    pub(crate) fn set_node(
        &mut self,
        node:Arc<Mutex<Node>>
    )
    {
        self.node=
            Some(node);
    }






    // -------------------------------------------------
    // Shape operations
    // -------------------------------------------------



    pub fn reshape(
        mut self,
        new_shape:&[usize],
    )
        -> Self
    {

        assert_eq!(
            self.numel(),
            new_shape.iter().product()
        );


        self.shape =
            Shape::from_slice(new_shape);


        self
    }




    pub fn flatten(
        self
    )
        -> Self
    {

        let size =
            self.numel();


        self.reshape(
            &[size]
        )
    }







    // -------------------------------------------------
    // Tensor cloning
    // -------------------------------------------------



    pub fn detach(
        self
    )
        -> Self
    {

        Self {

            storage:
                self.storage,

            shape:
                self.shape,

            dtype:
                self.dtype,

            device:
                self.device,

            requires_grad:false,

            grad:None,

            node:None,
        }
    }




    pub fn requires_grad_tensor(
        mut self
    )
        -> Self
    {

        self.requires_grad=true;

        self
    }




    // -------------------------------------------------
    // Internal helpers
    // -------------------------------------------------



    pub(crate) fn empty_like(
        other:&Tensor
    )
        -> Tensor
    {

        Tensor::zeros(
            other.shape.dims(),
            other.dtype,
            other.device,
        )
    }




    pub(crate) fn same_shape(
        a:&Tensor,
        b:&Tensor,
    )
    {
        assert_eq!(
            a.shape,
            b.shape,
            "Tensor shape mismatch"
        );
    }





    pub(crate) fn assert_float(
        &self
    )
    {

        match self.dtype {

            DType::Float32 |
            DType::Float64 => {},


            _ =>
                panic!(
                    "Operation requires floating tensor"
                ),
        }
    }


}


impl Tensor {


    // -------------------------------------------------
    // Device movement
    // -------------------------------------------------


    /// Move tensor to another device.
    ///
    /// Actual memory transfer is implemented by device backends.
    pub fn to(
        mut self,
        device: Device,
    ) -> Self {

        self.device = device;

        self
    }




    /// Check whether tensor is on CPU.
    pub fn is_cpu(
        &self
    ) -> bool {

        matches!(
            self.device,
            Device::CPU
        )
    }




    /// Check whether tensor is on GPU.
    pub fn is_gpu(
        &self
    ) -> bool {

        matches!(
            self.device,
            Device::GPU(_)
        )
    }






    // -------------------------------------------------
    // Element access
    // -------------------------------------------------


    pub fn get_f32(
        &self,
        index:usize,
    )
        -> f32
    {

        match &self.storage {

            Storage::F32(data)=>
                data[index],


            _ =>
                panic!(
                    "Tensor is not Float32"
                ),
        }
    }



    pub fn get_f64(
        &self,
        index:usize,
    )
        -> f64
    {

        match &self.storage {

            Storage::F64(data)=>
                data[index],


            _ =>
                panic!(
                    "Tensor is not Float64"
                ),
        }
    }




    pub fn set_f32(
        &mut self,
        index:usize,
        value:f32,
    )
    {

        match &mut self.storage {

            Storage::F32(data)=>
                data[index]=value,


            _ =>
                panic!(
                    "Tensor is not Float32"
                ),
        }
    }




    pub fn set_f64(
        &mut self,
        index:usize,
        value:f64,
    )
    {

        match &mut self.storage {

            Storage::F64(data)=>
                data[index]=value,


            _ =>
                panic!(
                    "Tensor is not Float64"
                ),
        }
    }






    // -------------------------------------------------
    // Internal constructors
    // -------------------------------------------------



    pub(crate) fn from_storage(
        storage:Storage,
        shape:Shape,
        dtype:DType,
        device:Device,
    )
        -> Self
    {

        assert_eq!(
            storage.len(),
            shape.numel()
        );


        Self {

            storage,

            shape,

            dtype,

            device,

            requires_grad:false,

            grad:None,

            node:None,
        }
    }





    pub(crate) fn with_gradient(
        mut self,
    )
        -> Self
    {

        self.requires_grad=true;

        self
    }



    pub(crate) fn accumulate_grad(
        &mut self,
        grad:Tensor,
    )
    {

        match &mut self.grad {


            Some(existing)=> {

                match (
                    existing.storage_mut(),
                    grad.storage()
                ){

                    (
                        Storage::F32(a),
                        Storage::F32(b)
                    )=>{

                        for i in 0..a.len(){
                            a[i]+=b[i];
                        }

                    }



                    (
                        Storage::F64(a),
                        Storage::F64(b)
                    )=>{

                        for i in 0..a.len(){
                            a[i]+=b[i];
                        }

                    }


                    _ =>
                        panic!(
                            "Gradient dtype mismatch"
                        )
                }
            }



            None => {

                self.grad=
                    Some(Box::new(grad));

            }

        }

    }




}




// =====================================================
// Trait Implementations
// =====================================================



impl Clone for Tensor {


    fn clone(
        &self
    )
        -> Self
    {

        Self {

            storage:
                self.storage.clone(),

            shape:
                self.shape.clone(),

            dtype:
                self.dtype,

            device:
                self.device,

            requires_grad:
                self.requires_grad,

            grad:
                self.grad.clone(),

            node:
                self.node.clone(),
        }

    }

}





impl fmt::Debug for Tensor {


    fn fmt(
        &self,
        f:&mut fmt::Formatter<'_>,
    )
        -> fmt::Result
    {

        f.debug_struct(
            "Tensor"
        )

        .field(
            "shape",
            &self.shape
        )

        .field(
            "dtype",
            &self.dtype
        )

        .field(
            "device",
            &self.device
        )

        .field(
            "requires_grad",
            &self.requires_grad
        )

        .finish()

    }

}





impl Storage {


    pub(crate) fn len(
        &self
    )
        -> usize
    {

        match self {

            Storage::F32(v)=>
                v.len(),


            Storage::F64(v)=>
                v.len(),


            Storage::I32(v)=>
                v.len(),


            Storage::I64(v)=>
                v.len(),

        }

    }

}
