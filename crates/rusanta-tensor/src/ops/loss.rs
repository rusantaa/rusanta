// rusanta-tensor/src/ops/loss.rs

//! Loss functions for machine learning.
//!
//! Used by:
//!
//! - Neural networks
//! - Regression models
//! - Classification models
//!
//! Implementations:
//!
//! - MSE
//! - MAE
//! - Binary Cross Entropy
//! - Cross Entropy
//! - Huber Loss
//! - KL Divergence


use crate::{
    Tensor,
    Result,
    TensorError,
};



use crate::tensor::Storage;


use crate::ops::math;







// =====================================================
// Mean Squared Error
// =====================================================



/// Mean Squared Error loss.
///
/// Formula:
///
/// ```text
/// MSE = mean((y - y_hat)^2)
/// ```
pub fn mse(
    prediction:&Tensor,
    target:&Tensor,
)
    -> Result<Tensor>
{


    Tensor::same_shape(
        prediction,
        target,
    );



    let diff =
        math::sub(
            prediction,
            target,
        )?;



    let squared =
        math::mul(
            &diff,
            &diff,
        )?;



    math::mean(
        &squared
    )

}









// =====================================================
// Mean Absolute Error
// =====================================================



/// Mean Absolute Error.
///
/// Formula:
///
/// ```text
/// MAE = mean(|y-y_hat|)
/// ```
pub fn mae(
    prediction:&Tensor,
    target:&Tensor,
)
    -> Result<Tensor>
{


    Tensor::same_shape(
        prediction,
        target,
    );



    let mut output =
        Tensor::empty_like(
            prediction
        );



    match (
        prediction.storage(),
        target.storage(),
        output.storage_mut(),
    )
    {


        (
            Storage::F32(pred),
            Storage::F32(real),
            Storage::F32(out),
        )=>{


            for i in 0..pred.len()
            {

                out[i]=
                    (pred[i]-real[i])
                    .abs();

            }


        }






        (
            Storage::F64(pred),
            Storage::F64(real),
            Storage::F64(out),
        )=>{


            for i in 0..pred.len()
            {

                out[i]=
                    (pred[i]-real[i])
                    .abs();

            }


        }





        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "MAE requires float tensors"
                    .into()
                )
            );

        }

    }



    math::mean(
        &output
    )

}









// =====================================================
// Binary Cross Entropy
// =====================================================



/// Binary Cross Entropy.
///
/// Formula:
///
/// ```text
///
/// -(y log(p)
/// +
/// (1-y)log(1-p))
///
/// ```
pub fn binary_cross_entropy(
    prediction:&Tensor,
    target:&Tensor,
)
    -> Result<Tensor>
{


    Tensor::same_shape(
        prediction,
        target,
    );



    let eps =
        1e-12;



    let clipped =
        clip(
            prediction,
            eps,
            1.0-eps,
        )?;





    let log_p =
        math::log(
            &clipped
        )?;





    let one_minus =
        math::sub(
            &Tensor::ones_like(
                prediction
            ),
            target,
        )?;





    let log_one_minus =
        math::log(
            &math::sub(
                &Tensor::ones_like(
                    prediction
                ),
                &clipped,
            )?
        )?;





    let first =
        math::mul(
            target,
            &log_p,
        )?;





    let second =
        math::mul(
            &one_minus,
            &log_one_minus,
        )?;





    let total =
        math::add(
            &first,
            &second,
        )?;





    let negative =
        math::mul_scalar(
            &total,
            -1.0,
        )?;





    math::mean(
        &negative
    )

}

// rusanta-tensor/src/ops/loss.rs
// Part 2/2



// =====================================================
// Cross Entropy
// =====================================================



/// Multi-class cross entropy loss.
///
/// Expected:
///
/// prediction:
/// ```text
/// [classes]
/// ```
///
/// target:
/// ```text
/// one-hot vector
/// ```
pub fn cross_entropy(
    prediction:&Tensor,
    target:&Tensor,
)
    -> Result<Tensor>
{


    Tensor::same_shape(
        prediction,
        target,
    );



    let eps =
        1e-12;



    let probabilities =
        clip(
            prediction,
            eps,
            1.0-eps,
        )?;



    let log_probs =
        math::log(
            &probabilities
        )?;



    let multiplied =
        math::mul(
            target,
            &log_probs,
        )?;



    let negative =
        math::mul_scalar(
            &multiplied,
            -1.0,
        )?;



    math::sum(
        &negative
    )

}









// =====================================================
// Softmax Cross Entropy
// =====================================================



/// Combined softmax + cross entropy.
///
/// More numerically stable than doing them separately.
pub fn softmax_cross_entropy(
    logits:&Tensor,
    target:&Tensor,
)
    -> Result<Tensor>
{


    let probabilities =
        crate::ops::nn::softmax(
            logits
        )?;



    cross_entropy(
        &probabilities,
        target,
    )

}









// =====================================================
// Huber Loss
// =====================================================



/// Huber loss.
///
/// Combines:
///
/// - MSE near zero
/// - MAE for large errors
///
/// Formula:
///
/// ```text
///
/// |x| <= delta:
///     0.5*x²
///
/// else:
///     delta*(|x|-0.5*delta)
///
/// ```
pub fn huber(
    prediction:&Tensor,
    target:&Tensor,
    delta:f64,
)
    -> Result<Tensor>
{


    Tensor::same_shape(
        prediction,
        target,
    );



    let mut output =
        Tensor::empty_like(
            prediction
        );



    match (
        prediction.storage(),
        target.storage(),
        output.storage_mut(),
    )
    {


        (
            Storage::F32(pred),
            Storage::F32(real),
            Storage::F32(out),
        )=>{


            let d =
                delta as f32;



            for i in 0..pred.len()
            {

                let error =
                    pred[i]-real[i];


                let abs =
                    error.abs();



                out[i] =
                    if abs <= d
                    {

                        0.5 *
                        error *
                        error

                    }
                    else
                    {

                        d *
                        (
                            abs
                            -
                            0.5*d
                        )

                    };

            }


        }






        (
            Storage::F64(pred),
            Storage::F64(real),
            Storage::F64(out),
        )=>{


            for i in 0..pred.len()
            {

                let error =
                    pred[i]-real[i];


                let abs =
                    error.abs();



                out[i] =
                    if abs <= delta
                    {

                        0.5 *
                        error *
                        error

                    }
                    else
                    {

                        delta *
                        (
                            abs
                            -
                            0.5*delta
                        )

                    };

            }


        }





        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "Huber requires float tensors"
                    .into()
                )
            );

        }


    }



    math::mean(
        &output
    )

}









// =====================================================
// KL Divergence
// =====================================================



/// Kullback-Leibler divergence.
///
/// Formula:
///
/// ```text
/// Σ p log(p/q)
/// ```
pub fn kl_divergence(
    prediction:&Tensor,
    target:&Tensor,
)
    -> Result<Tensor>
{


    Tensor::same_shape(
        prediction,
        target,
    );



    let eps =
        1e-12;



    let p =
        clip(
            target,
            eps,
            1.0,
        )?;



    let q =
        clip(
            prediction,
            eps,
            1.0,
        )?;





    let ratio =
        math::div(
            &p,
            &q,
        )?;





    let log_ratio =
        math::log(
            &ratio
        )?;





    let value =
        math::mul(
            &p,
            &log_ratio,
        )?;





    math::sum(
        &value
    )

}









// =====================================================
// Utility
// =====================================================



/// Clamp tensor values.
///
/// Used for numerical stability.
fn clip(
    x:&Tensor,
    min:f64,
    max:f64,
)
    -> Result<Tensor>
{


    let mut output =
        Tensor::empty_like(
            x
        );



    match (
        x.storage(),
        output.storage_mut(),
    )
    {


        (
            Storage::F32(input),
            Storage::F32(out),
        )=>{


            let lo =
                min as f32;


            let hi =
                max as f32;



            for i in 0..input.len()
            {

                out[i]=
                    input[i]
                    .max(lo)
                    .min(hi);

            }


        }





        (
            Storage::F64(input),
            Storage::F64(out),
        )=>{


            for i in 0..input.len()
            {

                out[i]=
                    input[i]
                    .max(min)
                    .min(max);

            }


        }





        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "clip requires float tensor"
                    .into()
                )
            );

        }


    }



    Ok(output)

}
