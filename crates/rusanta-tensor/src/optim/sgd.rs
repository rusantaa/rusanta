// rusanta-tensor/src/optim/sgd.rs

//! Stochastic Gradient Descent optimizer.
//!
//! Formula:
//!
//! ```text
//! w = w - lr * grad
//! ```
//!
//! Supports:
//!
//! - learning rate
//! - weight decay
//! - multiple parameters



use crate::Tensor;


use super::{
    Optimizer,
    apply_gradient,
    clear_parameter_grad,
};









// =====================================================
// SGD Optimizer
// =====================================================



pub struct SGD {


    parameters:
        Vec<Tensor>,



    learning_rate:
        f64,



    weight_decay:
        f64,


}









impl SGD {



    pub fn new(
        parameters:Vec<Tensor>,
        learning_rate:f64,
    )
        -> Self
    {

        Self {

            parameters,


            learning_rate,


            weight_decay:0.0,


        }

    }









    pub fn with_weight_decay(
        mut self,
        decay:f64,
    )
        -> Self
    {

        self.weight_decay =
            decay;


        self

    }









    pub fn learning_rate(
        &self,
    )
        -> f64
    {

        self.learning_rate

    }





}









impl Optimizer for SGD {



    fn step(
        &mut self,
    )
    {


        for parameter in
            self.parameters.iter_mut()
        {


            apply_gradient(
                parameter,
                self.learning_rate,
            );



            //
            // Weight decay:
            //
            // w -= lr * decay * w
            //
            if self.weight_decay > 0.0
            {

                apply_weight_decay(
                    parameter,
                    self.learning_rate,
                    self.weight_decay,
                );

            }


        }


    }








    fn zero_grad(
        &mut self,
    )
    {


        for parameter in
            self.parameters.iter_mut()
        {

            clear_parameter_grad(
                parameter
            );

        }


    }








    fn parameter_count(
        &self,
    )
        -> usize
    {

        self.parameters.len()

    }



}









// =====================================================
// Weight Decay
// =====================================================



fn apply_weight_decay(
    parameter:&mut Tensor,
    lr:f64,
    decay:f64,
)
{


    match parameter.storage_mut()
    {

        crate::tensor::Storage::F32(data)=>
        {

            let factor =
                (lr * decay)
                as f32;



            for value in data
            {

                *value -=
                    factor * *value;

            }

        }





        crate::tensor::Storage::F64(data)=>
        {


            let factor =
                lr * decay;



            for value in data
            {

                *value -=
                    factor * *value;

            }


        }





        _=>{}

    }


}
