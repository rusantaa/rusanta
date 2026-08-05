// rusanta-tensor/src/optim/rmsprop.rs

//! RMSProp optimizer.
//!
//! Root Mean Square Propagation.
//!
//! Designed for:
//!
//! - recurrent networks
//! - noisy optimization landscapes
//! - adaptive learning rate training



use crate::{
    Tensor,
    tensor::Storage,
};


use super::{
    Optimizer,
    clear_parameter_grad,
};









// =====================================================
// RMSProp
// =====================================================



pub struct RMSProp {


    parameters:
        Vec<Tensor>,



    square_average:
        Vec<Vec<f64>>,



    learning_rate:
        f64,



    alpha:
        f64,



    epsilon:
        f64,



}









impl RMSProp {



    pub fn new(
        parameters:Vec<Tensor>,
        learning_rate:f64,
    )
        -> Self
    {



        let square_average =

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


            square_average,


            learning_rate,


            alpha:
                0.99,


            epsilon:
                1e-8,


        }


    }









    pub fn alpha(
        mut self,
        value:f64,
    )
        -> Self
    {

        self.alpha =
            value;


        self

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









impl Optimizer for RMSProp {



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



            let average =
                &mut self.square_average[index];







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





                        average[i] =

                            self.alpha
                            *
                            average[i]

                            +

                            (
                                1.0-self.alpha
                            )
                            *
                            g*g;







                        param[i] -=

                            (
                                self.learning_rate
                                *
                                g
                                /
                                (
                                    average[i].sqrt()
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





                        average[i] =

                            self.alpha
                            *
                            average[i]

                            +

                            (
                                1.0-self.alpha
                            )
                            *
                            g*g;







                        param[i] -=

                            self.learning_rate
                            *
                            g
                            /
                            (
                                average[i].sqrt()
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
