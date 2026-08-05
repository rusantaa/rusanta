// rusanta-tensor/src/optim/mod.rs

//! Optimization algorithms for Rusanta.
//!
//! Optimizers update model parameters using gradients.
//!
//! Supported:
//!
//! - SGD
//! - Momentum
//! - Adam
//! - AdamW
//! - RMSProp
//! - AdaGrad
//!
//! Architecture inspired by:
//!
//! - PyTorch Optimizer
//! - TensorFlow Optimizers


use crate::Tensor;





pub mod sgd;

pub mod momentum;

pub mod adam;

pub mod adamw;

pub mod rmsprop;

pub mod adagrad;









// =====================================================
// Optimizer Trait
// =====================================================



/// Base optimizer interface.
pub trait Optimizer {



    /// Update parameters.
    fn step(
        &mut self,
    );






    /// Clear accumulated gradients.
    fn zero_grad(
        &mut self,
    );






    /// Number of optimized parameters.
    fn parameter_count(
        &self,
    )
        -> usize;


}









// =====================================================
// Parameter Container
// =====================================================



/// Trainable parameter wrapper.
///
/// Used by neural network modules.
pub struct Parameter {


    tensor:
        Tensor,



}



impl Parameter {



    pub fn new(
        tensor:Tensor,
    )
        -> Self
    {

        Self {

            tensor,

        }

    }






    pub fn tensor(
        &self,
    )
        -> &Tensor
    {

        &self.tensor

    }






    pub fn tensor_mut(
        &mut self,
    )
        -> &mut Tensor
    {

        &mut self.tensor

    }

}









// =====================================================
// Optimizer Utilities
// =====================================================



/// Apply gradient descent update.
///
/// Parameter:
///
/// ```text
/// w = w - lr * grad
/// ```
pub(crate) fn apply_gradient(
    parameter:&mut Tensor,
    learning_rate:f64,
)
{

    let gradient =
        match parameter.grad()
        {

            Some(g)=>
                g.clone(),


            None=>
                return,

        };



    match (
        parameter.storage_mut(),
        gradient.storage(),
    )
    {


        (
            crate::tensor::Storage::F32(param),
            crate::tensor::Storage::F32(grad),
        )=>{


            let lr =
                learning_rate as f32;



            for i in 0..param.len()
            {

                param[i]
                    -=
                    lr * grad[i];

            }

        }





        (
            crate::tensor::Storage::F64(param),
            crate::tensor::Storage::F64(grad),
        )=>{


            for i in 0..param.len()
            {

                param[i]
                    -=
                    learning_rate * grad[i];

            }

        }




        _=>{}

    }


}









/// Remove gradients from parameters.
pub(crate) fn clear_parameter_grad(
    parameter:&mut Tensor,
)
{

    parameter.clear_grad();

}
