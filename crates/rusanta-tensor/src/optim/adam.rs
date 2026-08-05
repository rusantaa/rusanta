// rusanta-tensor/src/optim/adam.rs

//! Adam optimizer.
//!
//! Adaptive Moment Estimation.
//!
//! Formula:
//!
//! m = β1*m + (1-β1)*g
//!
//! v = β2*v + (1-β2)*g²
//!
//! m̂ = m/(1-β1^t)
//!
//! v̂ = v/(1-β2^t)
//!
//! w = w - lr*m̂/(sqrt(v̂)+eps)



use crate::{
    Tensor,
    tensor::Storage,
};


use super::{
    Optimizer,
    clear_parameter_grad,
};









// =====================================================
// Adam Optimizer
// =====================================================



pub struct Adam {


    parameters:
        Vec<Tensor>,



    first_moment:
        Vec<Vec<f64>>,



    second_moment:
        Vec<Vec<f64>>,



    learning_rate:
        f64,



    beta1:
        f64,



    beta2:
        f64,



    epsilon:
        f64,



    timestep:
        usize,


}









impl Adam {



    pub fn new(
        parameters:Vec<Tensor>,
        learning_rate:f64,
    )
        -> Self
    {


        let first_moment =
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



        let second_moment =
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


            first_moment,


            second_moment,


            learning_rate,


            beta1:
                0.9,


            beta2:
                0.999,


            epsilon:
                1e-8,


            timestep:
                0,


        }


    }









    pub fn beta1(
        mut self,
        value:f64,
    )
        -> Self
    {

        self.beta1 =
            value;


        self

    }








    pub fn beta2(
        mut self,
        value:f64,
    )
        -> Self
    {

        self.beta2 =
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









impl Optimizer for Adam {



    fn step(
        &mut self,
    )
    {


        self.timestep += 1;



        let t =
            self.timestep as f64;





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




            let m =
                &mut self.first_moment[index];



            let v =
                &mut self.second_moment[index];





            let bias_correction1 =

                1.0
                -
                self.beta1.powf(t);




            let bias_correction2 =

                1.0
                -
                self.beta2.powf(t);







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



                        m[i] =

                            self.beta1*m[i]

                            +

                            (1.0-self.beta1)
                            *
                            g;



                        v[i] =

                            self.beta2*v[i]

                            +

                            (1.0-self.beta2)
                            *
                            g*g;



                        let m_hat =

                            m[i]
                            /
                            bias_correction1;



                        let v_hat =

                            v[i]
                            /
                            bias_correction2;



                        param[i] -=

                            (
                                self.learning_rate
                                *
                                m_hat
                                /
                                (
                                    v_hat.sqrt()
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



                        m[i] =

                            self.beta1*m[i]

                            +

                            (1.0-self.beta1)
                            *
                            g;



                        v[i] =

                            self.beta2*v[i]

                            +

                            (1.0-self.beta2)
                            *
                            g*g;



                        let m_hat =

                            m[i]
                            /
                            bias_correction1;



                        let v_hat =

                            v[i]
                            /
                            bias_correction2;



                        param[i] -=

                            self.learning_rate
                            *
                            m_hat
                            /
                            (
                                v_hat.sqrt()
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
