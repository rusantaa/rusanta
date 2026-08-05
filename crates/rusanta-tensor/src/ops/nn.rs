// rusanta-tensor/src/ops/nn.rs

//! Neural network tensor operations.
//!
//! Includes:
//!
//! - Activation functions
//! - Neural network primitives
//! - Layer operations
//!
//! Used by rusanta-ml models.

use crate::{
    Tensor,
    Result,
    TensorError,
};



use crate::tensor::Storage;






// =====================================================
// ReLU
// =====================================================



/// Rectified Linear Unit.
///
/// Formula:
///
/// ```text
/// f(x)=max(0,x)
/// ```
pub fn relu(
    x:&Tensor,
)
    -> Result<Tensor>
{


    let mut output =
        Tensor::empty_like(x);



    match (
        x.storage(),
        output.storage_mut(),
    )
    {


        (
            Storage::F32(input),
            Storage::F32(out),
        )=>{


            for i in 0..input.len()
            {

                out[i] =
                    if input[i] > 0.0 {

                        input[i]

                    }
                    else {

                        0.0

                    };

            }

        }





        (
            Storage::F64(input),
            Storage::F64(out),
        )=>{


            for i in 0..input.len()
            {

                out[i] =
                    if input[i] > 0.0 {

                        input[i]

                    }
                    else {

                        0.0

                    };

            }

        }




        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "relu requires floating tensor"
                    .into()
                )
            );

        }


    }



    Ok(output)

}









// =====================================================
// Sigmoid
// =====================================================



/// Sigmoid activation.
///
/// Formula:
///
/// ```text
/// 1/(1+e^-x)
/// ```
pub fn sigmoid(
    x:&Tensor,
)
    -> Result<Tensor>
{


    let mut output =
        Tensor::empty_like(x);



    match (
        x.storage(),
        output.storage_mut(),
    )
    {


        (
            Storage::F32(input),
            Storage::F32(out),
        )=>{


            for i in 0..input.len()
            {

                out[i] =
                    1.0 /
                    (
                        1.0 +
                        (-input[i]).exp()
                    );

            }

        }





        (
            Storage::F64(input),
            Storage::F64(out),
        )=>{


            for i in 0..input.len()
            {

                out[i] =
                    1.0 /
                    (
                        1.0 +
                        (-input[i]).exp()
                    );

            }

        }




        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "sigmoid requires floating tensor"
                    .into()
                )
            );

        }

    }



    Ok(output)

}









// =====================================================
// Tanh
// =====================================================



/// Hyperbolic tangent activation.
///
/// Formula:
///
/// ```text
/// tanh(x)
/// ```
pub fn tanh(
    x:&Tensor,
)
    -> Result<Tensor>
{


    let mut output =
        Tensor::empty_like(x);



    match (
        x.storage(),
        output.storage_mut(),
    )
    {


        (
            Storage::F32(input),
            Storage::F32(out),
        )=>{


            for i in 0..input.len()
            {

                out[i] =
                    input[i].tanh();

            }

        }





        (
            Storage::F64(input),
            Storage::F64(out),
        )=>{


            for i in 0..input.len()
            {

                out[i] =
                    input[i].tanh();

            }

        }




        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "tanh requires floating tensor"
                    .into()
                )
            );

        }

    }



    Ok(output)

}



use crate::ops::math;







// =====================================================
// Softmax
// =====================================================



/// Softmax activation.
///
/// Usually used for classification output.
///
/// Formula:
///
/// ```text
/// exp(x_i) / Σ exp(x)
/// ```
pub fn softmax(
    x:&Tensor,
)
    -> Result<Tensor>
{


    let mut output =
        Tensor::empty_like(x);



    match (
        x.storage(),
        output.storage_mut(),
    )
    {


        (
            Storage::F32(input),
            Storage::F32(out),
        )=>{


            let mut sum =
                0.0f32;



            for value in input {

                sum += value.exp();

            }



            for i in 0..input.len()
            {

                out[i] =
                    input[i].exp()
                    /
                    sum;

            }


        }





        (
            Storage::F64(input),
            Storage::F64(out),
        )=>{


            let mut sum =
                0.0f64;



            for value in input {

                sum += value.exp();

            }



            for i in 0..input.len()
            {

                out[i] =
                    input[i].exp()
                    /
                    sum;

            }


        }




        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "softmax requires floating tensor"
                    .into()
                )
            );

        }

    }



    Ok(output)

}









// =====================================================
// Log Softmax
// =====================================================



/// LogSoftmax activation.
///
/// Formula:
///
/// ```text
/// log(softmax(x))
/// ```
pub fn log_softmax(
    x:&Tensor,
)
    -> Result<Tensor>
{


    let sm =
        softmax(x)?;


    math::log(
        &sm
    )

}









// =====================================================
// Linear Layer
// =====================================================



/// Fully connected layer.
///
/// Computes:
///
/// ```text
/// y = xW + b
///
/// x : [batch, input]
/// W : [input, output]
/// b : [output]
/// ```
pub fn linear(
    input:&Tensor,
    weight:&Tensor,
    bias:Option<&Tensor>,
)
    -> Result<Tensor>
{


    let mut output =
        math::matmul(
            input,
            weight,
        )?;





    if let Some(b)=bias
    {

        output =
            add_bias(
                &output,
                b,
            )?;

    }



    Ok(output)

}









// =====================================================
// Bias Addition
// =====================================================



fn add_bias(
    x:&Tensor,
    bias:&Tensor,
)
    -> Result<Tensor>
{


    if bias.ndim()!=1
    {

        return Err(
            TensorError::ShapeMismatch {
                expected:
                    "[features]".into(),

                got:
                    format!(
                        "{:?}",
                        bias.shape()
                    ),
            }
        );

    }




    let mut output =
        Tensor::empty_like(x);




    match (
        x.storage(),
        bias.storage(),
        output.storage_mut(),
    )
    {


        (
            Storage::F32(data),
            Storage::F32(b),
            Storage::F32(out),
        )=>{


            let features =
                b.len();



            for i in 0..data.len()
            {

                out[i]=
                    data[i]
                    +
                    b[i % features];

            }


        }






        (
            Storage::F64(data),
            Storage::F64(b),
            Storage::F64(out),
        )=>{


            let features =
                b.len();



            for i in 0..data.len()
            {

                out[i]=
                    data[i]
                    +
                    b[i % features];

            }


        }





        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "bias addition requires float tensors"
                    .into()
                )
            );

        }

    }



    Ok(output)

}



use crate::tensor::Shape;






// =====================================================
// Dropout
// =====================================================



/// Dropout regularization.
///
/// During training:
///
/// ```text
/// y = x * mask / (1-p)
/// ```
///
/// Current implementation:
///
/// - deterministic scaling
/// - random mask generation will be added
///   through RNG subsystem
pub fn dropout(
    x:&Tensor,
    probability:f64,
    training:bool,
)
    -> Result<Tensor>
{


    if !training
    {

        return Ok(
            x.clone()
        );

    }



    if probability < 0.0
        || probability >= 1.0
    {

        return Err(
            TensorError::UnsupportedOperation(
                "dropout probability must be between 0 and 1"
                .into()
            )
        );

    }



    let scale =
        1.0 /
        (1.0 - probability);



    crate::ops::math::mul_scalar(
        x,
        scale,
    )

}









// =====================================================
// Layer Normalization
// =====================================================



/// Layer normalization.
///
/// Formula:
///
/// ```text
///
/// y = (x-mean)/sqrt(var+eps)
///
/// ```
pub fn layer_norm(
    x:&Tensor,
    eps:f64,
)
    -> Result<Tensor>
{


    let mean =
        crate::ops::math::mean(
            x
        )?;



    let centered =
        crate::ops::math::sub(
            x,
            &mean,
        )?;





    //
    // Variance approximation:
    //
    // var = mean((x-mean)^2)
    //
    let squared =
        crate::ops::math::mul(
            &centered,
            &centered,
        )?;




    let variance =
        crate::ops::math::mean(
            &squared
        )?;





    let variance_eps =
        crate::ops::math::add_scalar(
            &variance,
            eps,
        )?;





    let std =
        sqrt(
            &variance_eps
        )?;





    crate::ops::math::div(
        &centered,
        &std,
    )

}









// =====================================================
// Batch Normalization Helper
// =====================================================



/// Simple batch normalization.
///
/// Formula:
///
/// ```text
/// y = (x-mean)/sqrt(var+eps)
/// ```
pub fn batch_norm(
    x:&Tensor,
    eps:f64,
)
    -> Result<Tensor>
{


    layer_norm(
        x,
        eps,
    )

}









// =====================================================
// Square Root Helper
// =====================================================



fn sqrt(
    x:&Tensor,
)
    -> Result<Tensor>
{


    let mut output =
        Tensor::empty_like(x);



    match (
        x.storage(),
        output.storage_mut(),
    )
    {


        (
            Storage::F32(input),
            Storage::F32(out),
        )=>{


            for i in 0..input.len()
            {

                out[i]=
                    input[i].sqrt();

            }


        }





        (
            Storage::F64(input),
            Storage::F64(out),
        )=>{


            for i in 0..input.len()
            {

                out[i]=
                    input[i].sqrt();

            }


        }





        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "sqrt requires float tensor"
                    .into()
                )
            );

        }


    }



    Ok(output)

}
