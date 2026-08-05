// rusanta-tensor/src/optim/momentum.rs

//! Momentum based stochastic gradient descent.
//!
//! Adds velocity accumulation to SGD.
//!
//! Supports:
//!
//! - Momentum
//! - Dampening
//! - Nesterov acceleration



use crate::{
    Tensor,
    tensor::Storage,
};


use super::{
    Optimizer,
    clear_parameter_grad,
};









// =====================================================
// Momentum Optimizer
// =====================================================



pub struct MomentumSGD {


    parameters:
        Vec<Tensor>,



    velocity:
        Vec<Vec<f64>>,



    learning_rate:
        f64,



    momentum:
        f64,



    dampening:
        f64,



    nesterov:
        bool,


}









impl MomentumSGD {



    pub fn new(
        parameters:Vec<Tensor>,
        learning_rate:f64,
        momentum:f64,
    )
        -> Self
    {


        let velocity =
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


            velocity,


            learning_rate,


            momentum,


            dampening:
                0.0,


            nesterov:
                false,


        }

    }









    pub fn dampening(
        mut self,
        value:f64,
    )
        -> Self
    {

        self.dampening =
            value;


        self

    }








    pub fn nesterov(
        mut self,
        enabled:bool,
    )
        -> Self
    {

        self.nesterov =
            enabled;


        self

    }



}









impl Optimizer for MomentumSGD {



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



            let velocity =
                &mut self.velocity[index];





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


                        velocity[i] =

                            self.momentum
                            *
                            velocity[i]

                            +

                            (1.0-self.dampening)
                            *
                            grad[i]
                                as f64;



                        let update =
                            if self.nesterov
                            {

                                self.momentum
                                *
                                velocity[i]

                                +

                                grad[i]
                                    as f64

                            }
                            else
                            {

                                velocity[i]

                            };



                        param[i]
                            -=
                            self.learning_rate
                            *
                            update
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


                        velocity[i] =

                            self.momentum
                            *
                            velocity[i]

                            +

                            (1.0-self.dampening)
                            *
                            grad[i];



                        let update =
                            if self.nesterov
                            {

                                self.momentum
                                *
                                velocity[i]

                                +

                                grad[i]

                            }
                            else
                            {

                                velocity[i]

                            };



                        param[i]
                            -=
                            self.learning_rate
                            *
                            update;


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
