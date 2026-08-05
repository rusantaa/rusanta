// rusanta-tensor/src/tensor.rs

//! Core Tensor implementation for Rusanta.
//!
//! Tensor is the fundamental object for:
//!
//! - numerical computation
//! - automatic differentiation
//! - neural networks
//! - optimization
//! - GPU execution
//!
//! Inspired by:
//!
//! - PyTorch Tensor
//! - NumPy ndarray
//! - JAX DeviceArray


use std::fmt;

use std::sync::{
    Arc,
    Mutex,
};


use crate::device::Device;


use crate::autograd::node::Node;







// =====================================================
// Data Type
// =====================================================



/// Tensor element type.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq
)]
pub enum DType {


    Float32,


    Float64,


    Int32,


    Int64,

}









impl DType {


    pub fn size_of(
        &self
    )
        -> usize
    {

        match self {

            DType::Float32 =>
                std::mem::size_of::<f32>(),


            DType::Float64 =>
                std::mem::size_of::<f64>(),


            DType::Int32 =>
                std::mem::size_of::<i32>(),


            DType::Int64 =>
                std::mem::size_of::<i64>(),

        }

    }





    pub fn is_float(
        &self
    )
        -> bool
    {

        matches!(
            self,
            DType::Float32
            |
            DType::Float64
        )

    }

}









// =====================================================
// Storage
// =====================================================



/// Raw tensor memory.
///
/// Future backends:
///
/// - CPU allocator
/// - CUDA memory
/// - WGPU buffers
/// - unified memory
#[derive(
    Debug,
    Clone
)]
pub enum Storage {


    F32(Vec<f32>),


    F64(Vec<f64>),


    I32(Vec<i32>),


    I64(Vec<i64>),

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




    pub fn dtype(
        &self
    )
        -> DType
    {

        match self {


            Storage::F32(_)=>
                DType::Float32,


            Storage::F64(_)=>
                DType::Float64,


            Storage::I32(_)=>
                DType::Int32,


            Storage::I64(_)=>
                DType::Int64,

        }

    }


}









// =====================================================
// Shape
// =====================================================



/// Tensor dimensional information.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq
)]
pub struct Shape {


    dims:
        Vec<usize>,


}







impl Shape {



    pub fn new(
        dims:Vec<usize>
    )
        -> Self
    {

        Self {

            dims,

        }

    }






    pub fn from_slice(
        dims:&[usize]
    )
        -> Self
    {

        Self {

            dims:
                dims.to_vec(),

        }

    }






    pub fn dims(
        &self
    )
        -> &[usize]
    {

        &self.dims

    }






    pub fn ndim(
        &self
    )
        -> usize
    {

        self.dims.len()

    }






    pub fn numel(
        &self
    )
        -> usize
    {

        if self.dims.is_empty()
        {

            return 0;

        }



        self.dims
            .iter()
            .product()

    }


}









// =====================================================
// Tensor
// =====================================================



/// Main tensor object.
///
/// Contains:
///
/// - data
/// - shape
/// - dtype
/// - device
/// - gradient
/// - autograd graph node
pub struct Tensor {


    storage:
        Storage,



    shape:
        Shape,



    dtype:
        DType,



    device:
        Device,




    requires_grad:
        bool,




    grad:
        Option<Box<Tensor>>,




    node:
        Option<Arc<Mutex<Node>>>,



}









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









impl Tensor {


    /// Create tensor from storage.
    pub fn from_storage(
        storage:Storage,
        shape:Shape,
        device:Device,
    )
        -> Self
    {


        let dtype =
            storage.dtype();



        assert_eq!(
            storage.len(),
            shape.numel(),
            "Storage size does not match shape"
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



impl Tensor {



    // =====================================================
    // Constructors
    // =====================================================



    /// Create tensor filled with zeros.
    pub fn zeros(
        shape:&[usize],
        dtype:DType,
        device:Device,
    )
        -> Self
    {


        let size =
            shape
                .iter()
                .product::<usize>();



        let storage =
            match dtype
            {

                DType::Float32 =>
                    Storage::F32(
                        vec![
                            0.0;
                            size
                        ]
                    ),



                DType::Float64 =>
                    Storage::F64(
                        vec![
                            0.0;
                            size
                        ]
                    ),



                DType::Int32 =>
                    Storage::I32(
                        vec![
                            0;
                            size
                        ]
                    ),



                DType::Int64 =>
                    Storage::I64(
                        vec![
                            0;
                            size
                        ]
                    ),

            };



        Self::from_storage(
            storage,
            Shape::from_slice(shape),
            device,
        )

    }








    /// Create tensor filled with ones.
    pub fn ones(
        shape:&[usize],
        dtype:DType,
        device:Device,
    )
        -> Self
    {


        let size =
            shape
                .iter()
                .product::<usize>();



        let storage =
            match dtype
            {

                DType::Float32 =>
                    Storage::F32(
                        vec![
                            1.0;
                            size
                        ]
                    ),



                DType::Float64 =>
                    Storage::F64(
                        vec![
                            1.0;
                            size
                        ]
                    ),



                DType::Int32 =>
                    Storage::I32(
                        vec![
                            1;
                            size
                        ]
                    ),



                DType::Int64 =>
                    Storage::I64(
                        vec![
                            1;
                            size
                        ]
                    ),

            };



        Self::from_storage(
            storage,
            Shape::from_slice(shape),
            device,
        )

    }








    /// Allocate tensor without initialization.
    ///
    /// Rust Vec initializes memory,
    /// but this API allows future allocator replacement.
    pub fn empty(
        shape:&[usize],
        dtype:DType,
        device:Device,
    )
        -> Self
    {


        Self::zeros(
            shape,
            dtype,
            device,
        )

    }









    /// Create zero tensor with same metadata.
    pub fn zeros_like(
        other:&Tensor,
    )
        -> Self
    {

        Self::zeros(
            other.shape.dims(),
            other.dtype,
            other.device,
        )

    }








    /// Create one tensor with same metadata.
    pub fn ones_like(
        other:&Tensor,
    )
        -> Self
    {

        Self::ones(
            other.shape.dims(),
            other.dtype,
            other.device,
        )

    }








    /// Create empty tensor matching another tensor.
    pub fn empty_like(
        other:&Tensor,
    )
        -> Self
    {

        Self::empty(
            other.shape.dims(),
            other.dtype,
            other.device,
        )

    }









    // =====================================================
    // Accessors
    // =====================================================



    pub fn storage(
        &self
    )
        -> &Storage
    {

        &self.storage

    }






    pub fn storage_mut(
        &mut self
    )
        -> &mut Storage
    {

        &mut self.storage

    }






    pub fn shape(
        &self
    )
        -> &Shape
    {

        &self.shape

    }






    pub fn dtype(
        &self
    )
        -> DType
    {

        self.dtype

    }






    pub fn device(
        &self
    )
        -> Device
    {

        self.device

    }






    pub fn ndim(
        &self
    )
        -> usize
    {

        self.shape.ndim()

    }






    pub fn numel(
        &self
    )
        -> usize
    {

        self.shape.numel()

    }



}



impl Tensor {



    // =====================================================
    // Autograd
    // =====================================================



    /// Enable or disable gradient tracking.
    pub fn set_requires_grad(
        &mut self,
        value:bool,
    )
    {

        self.requires_grad =
            value;

    }







    /// Check if tensor participates
    /// in automatic differentiation.
    pub fn requires_grad(
        &self,
    )
        -> bool
    {

        self.requires_grad

    }








    /// Attach computation graph node.
    pub(crate) fn set_node(
        &mut self,
        node:Arc<Mutex<Node>>,
    )
    {

        self.node =
            Some(node);

    }








    /// Get computation graph node.
    pub fn node(
        &self,
    )
        -> Option<Arc<Mutex<Node>>>
    {

        self.node.clone()

    }









    // =====================================================
    // Gradient Handling
    // =====================================================



    /// Check whether gradient exists.
    pub fn has_grad(
        &self,
    )
        -> bool
    {

        self.grad.is_some()

    }








    /// Get gradient tensor.
    pub fn grad(
        &self,
    )
        -> Option<&Tensor>
    {

        self.grad
            .as_deref()

    }








    /// Mutable gradient access.
    pub fn grad_mut(
        &mut self,
    )
        -> Option<&mut Tensor>
    {

        self.grad
            .as_deref_mut()

    }








    /// Replace gradient.
    pub fn set_grad(
        &mut self,
        gradient:Tensor,
    )
    {

        self.grad =
            Some(
                Box::new(
                    gradient
                )
            );

    }








    /// Remove stored gradient.
    pub fn clear_grad(
        &mut self,
    )
    {

        self.grad =
            None;

    }









    /// Accumulate gradient.
    ///
    /// Used by backward engine.
    pub fn accumulate_grad(
        &mut self,
        gradient:&Tensor,
    )
    {

        match &mut self.grad
        {


            Some(existing)=>
            {

                accumulate_storage(
                    &mut existing.storage,
                    &gradient.storage,
                );


            }




            None=>
            {

                self.grad =
                    Some(
                        Box::new(
                            gradient.clone()
                        )
                    );


            }


        }


    }









    // =====================================================
    // Tensor Properties
    // =====================================================



    pub fn is_leaf(
        &self,
    )
        -> bool
    {

        self.node.is_none()

    }





    pub fn detach(
        &self,
    )
        -> Tensor
    {

        let mut output =
            self.clone();



        output.requires_grad =
            false;



        output.node =
            None;



        output.grad =
            None;



        output

    }



}









// =====================================================
// Gradient Storage Accumulation
// =====================================================



fn accumulate_storage(
    a:&mut Storage,
    b:&Storage,
)
{


    match (
        a,
        b,
    )
    {



        (
            Storage::F32(x),
            Storage::F32(y),
        )=>{


            for i in 0..x.len()
            {

                x[i]+=y[i];

            }


        }






        (
            Storage::F64(x),
            Storage::F64(y),
        )=>{


            for i in 0..x.len()
            {

                x[i]+=y[i];

            }


        }






        _=>{

            // Integer tensors
            // do not accumulate gradients.

        }


    }


}



impl Tensor {



    // =====================================================
    // Validation
    // =====================================================



    /// Ensure two tensors have the same shape.
    pub fn same_shape(
        a:&Tensor,
        b:&Tensor,
    )
    {

        assert_eq!(
            a.shape,
            b.shape,
            "Tensor shape mismatch: {:?} vs {:?}",
            a.shape,
            b.shape,
        );

    }








    /// Ensure tensor contains floating data.
    pub fn assert_float(
        &self,
    )
    {

        if !self.dtype.is_float()
        {

            panic!(
                "Operation requires floating point tensor, got {:?}",
                self.dtype
            );

        }

    }








    /// Returns true if tensor is empty.
    pub fn is_empty(
        &self,
    )
        -> bool
    {

        self.numel()==0

    }








    /// Return number of elements.
    pub fn size(
        &self,
    )
        -> usize
    {

        self.numel()

    }



}









// =====================================================
// Debug
// =====================================================



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
            &self.shape,
        )
        .field(
            "dtype",
            &self.dtype,
        )
        .field(
            "device",
            &self.device,
        )
        .field(
            "requires_grad",
            &self.requires_grad,
        )
        .field(
            "has_grad",
            &self.has_grad(),
        )
        .finish()

    }


}









// =====================================================
// Display
// =====================================================



impl fmt::Display for Tensor {


    fn fmt(
        &self,
        f:&mut fmt::Formatter<'_>,
    )
        -> fmt::Result
    {


        writeln!(
            f,
            "Tensor("
        )?;


        writeln!(
            f,
            "  shape: {:?},",
            self.shape.dims()
        )?;


        writeln!(
            f,
            "  dtype: {:?},",
            self.dtype
        )?;


        writeln!(
            f,
            "  device: {:?},",
            self.device
        )?;


        writeln!(
            f,
            "  requires_grad: {}",
            self.requires_grad
        )?;


        write!(
            f,
            ")"
        )

    }


}









// =====================================================
// Convenience Conversions
// =====================================================



impl Tensor {


    /// Convert tensor into vector if type matches.
    pub fn to_vec_f32(
        &self,
    )
        -> Option<Vec<f32>>
    {


        match &self.storage
        {

            Storage::F32(v)=>
                Some(v.clone()),


            _=>
                None,

        }

    }








    pub fn to_vec_f64(
        &self,
    )
        -> Option<Vec<f64>>
    {

        match &self.storage
        {

            Storage::F64(v)=>
                Some(v.clone()),


            _=>
                None,

        }

    }



}
