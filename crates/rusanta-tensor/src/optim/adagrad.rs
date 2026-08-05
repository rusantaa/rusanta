// rusanta-tensor/src/optim/adagrad.rs

//! AdaGrad optimizer.
//!
//! Adaptive Gradient Algorithm.
//!
//! Features:
//!
//! - individual learning rates
//! - accumulated gradient history
//! - sparse optimization friendly
//!
//! Commonly used for:
//!
//! - sparse features
//! - NLP models
//! - recommendation systems



use crate::{
    Tensor,
    tensor::Storage,
};


use super::{
    Optimizer,
    clear_parameter_grad,
};









// =====================================================
// AdaGrad
// =====================================================



pub struct AdaGrad {


    parameters:
        Vec<Tensor>,



    accumulated_gradient:
        Vec<Vec<f64>>,



    learning_rate:
        f64,



    epsilon:
        f64,


}









impl AdaGrad {



    pub fn new(
        parameters:Vec<Tensor>,
        learning_rate:f64,
    )
        -> Self
    {


        let accumulated_gradient =

            parameters
                .iter()
                .map(
                    |p|
                    {

                        vec![
                            0.0;
                            p.numel()
                        ]

                    }
                )
                .collect();



        Self {


            parameters,


            accumulated_gradient,


            learning_rate,


            epsilon:
                1e-8,


        }


    }









    pub fn epsilon(
        mut self,
        value:f64,
    )
        -> Self
    {

        self.epsilon =
            value;


        self

    }



}









impl Optimizer for AdaGrad {



    fn step(
        &mut self,
    )
    {


        for (
            index,
            parameter
        )
        in
            self.parameters
                .iter_mut()
                .enumerate()
        {


            let gradient =
                match parameter.grad()
                {

                    Some(g)=>
                        g.clone(),


                    None=>
                        continue,

                };



            let accumulator =
                &mut self.accumulated_gradient[index];







            match (
                parameter.storage_mut(),
                gradient.storage(),
            )
            {



                (
                    Storage::F32(param),
                    Storage::F32(grad),
                )=>
                {



                    for i in 0..param.len()
                    {



                        let g =
                            grad[i]
                            as f64;





                        accumulator[i]
                            +=
                            g*g;







                        param[i] -=

                            (
                                self.learning_rate
                                *
                                g
                                /
                                (
                                    accumulator[i]
                                    .sqrt()
                                    +
                                    self.epsilon
                                )
                            )
                            as f32;


                    }


                }









                (
                    Storage::F64(param),
                    Storage::F64(grad),
                )=>
                {



                    for i in 0..param.len()
                    {



                        let g =
                            grad[i];





                        accumulator[i]
                            +=
                            g*g;







                        param[i] -=

                            self.learning_rate
                            *
                            g
                            /
                            (
                                accumulator[i]
                                .sqrt()
                                +
                                self.epsilon
                            );


                    }


                }









                _=>{}

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
