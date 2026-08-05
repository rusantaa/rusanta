// rusanta-tensor/src/ops/math.rs

//! Mathematical tensor operations.
//!
//! Includes:
//!
//! - arithmetic
//! - scalar operations
//! - reductions
//! - matrix operations
//!
//! Every differentiable operation can attach
//! an autograd Node.


use std::sync::{
    Arc,
    Mutex,
};



use crate::{
    Tensor,
    Result,
    TensorError,
    Device,
};



use crate::autograd::node::{
    Node,
    Op,
};






// =====================================================
// Addition
// =====================================================



pub fn add(
    a:&Tensor,
    b:&Tensor,
)
    -> Result<Tensor>
{


    Tensor::same_shape(
        a,
        b
    );



    let mut output =
        Tensor::empty_like(a);



    match (
        a.storage(),
        b.storage(),
        output.storage_mut(),
    )
    {


        (
            crate::tensor::Storage::F32(x),
            crate::tensor::Storage::F32(y),
            crate::tensor::Storage::F32(out),
        )=>{

            for i in 0..x.len()
            {

                out[i]=
                    x[i]+y[i];

            }

        }




        (
            crate::tensor::Storage::F64(x),
            crate::tensor::Storage::F64(y),
            crate::tensor::Storage::F64(out),
        )=>{

            for i in 0..x.len()
            {

                out[i]=
                    x[i]+y[i];

            }

        }



        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "add requires matching float tensors"
                        .into()
                )
            );

        }


    }





    if a.requires_grad()
        || b.requires_grad()
    {

        output =
            attach_binary_node(
                output,
                a,
                b,
                Op::Add,
            );

    }



    Ok(output)

}









// =====================================================
// Subtraction
// =====================================================



pub fn sub(
    a:&Tensor,
    b:&Tensor,
)
    -> Result<Tensor>
{


    Tensor::same_shape(
        a,
        b
    );



    let mut output =
        Tensor::empty_like(a);



    match (
        a.storage(),
        b.storage(),
        output.storage_mut(),
    )
    {


        (
            crate::tensor::Storage::F32(x),
            crate::tensor::Storage::F32(y),
            crate::tensor::Storage::F32(out),
        )=>{

            for i in 0..x.len()
            {

                out[i]=
                    x[i]-y[i];

            }

        }




        (
            crate::tensor::Storage::F64(x),
            crate::tensor::Storage::F64(y),
            crate::tensor::Storage::F64(out),
        )=>{

            for i in 0..x.len()
            {

                out[i]=
                    x[i]-y[i];

            }

        }



        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "sub requires matching float tensors"
                        .into()
                )
            );

        }


    }





    if a.requires_grad()
        || b.requires_grad()
    {

        output =
            attach_binary_node(
                output,
                a,
                b,
                Op::Sub,
            );

    }



    Ok(output)

}









// =====================================================
// Multiplication
// =====================================================



pub fn mul(
    a:&Tensor,
    b:&Tensor,
)
    -> Result<Tensor>
{


    Tensor::same_shape(
        a,
        b
    );



    let mut output =
        Tensor::empty_like(a);



    match (
        a.storage(),
        b.storage(),
        output.storage_mut(),
    )
    {


        (
            crate::tensor::Storage::F32(x),
            crate::tensor::Storage::F32(y),
            crate::tensor::Storage::F32(out),
        )=>{

            for i in 0..x.len()
            {

                out[i]=
                    x[i]*y[i];

            }

        }




        (
            crate::tensor::Storage::F64(x),
            crate::tensor::Storage::F64(y),
            crate::tensor::Storage::F64(out),
        )=>{

            for i in 0..x.len()
            {

                out[i]=
                    x[i]*y[i];

            }

        }



        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "mul requires matching float tensors"
                        .into()
                )
            );

        }


    }





    if a.requires_grad()
        || b.requires_grad()
    {

        output =
            attach_binary_node(
                output,
                a,
                b,
                Op::Mul,
            );

    }



    Ok(output)

}



// =====================================================
// Division
// =====================================================



pub fn div(
    a:&Tensor,
    b:&Tensor,
)
    -> Result<Tensor>
{

    Tensor::same_shape(
        a,
        b
    );


    let mut output =
        Tensor::empty_like(a);



    match (
        a.storage(),
        b.storage(),
        output.storage_mut(),
    )
    {


        (
            crate::tensor::Storage::F32(x),
            crate::tensor::Storage::F32(y),
            crate::tensor::Storage::F32(out),
        )=>{

            for i in 0..x.len()
            {

                out[i]=
                    x[i] / y[i];

            }

        }




        (
            crate::tensor::Storage::F64(x),
            crate::tensor::Storage::F64(y),
            crate::tensor::Storage::F64(out),
        )=>{

            for i in 0..x.len()
            {

                out[i]=
                    x[i] / y[i];

            }

        }



        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "div requires matching float tensors"
                    .into()
                )
            );

        }

    }



    Ok(output)

}









// =====================================================
// Negation
// =====================================================



pub fn neg(
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
            crate::tensor::Storage::F32(input),
            crate::tensor::Storage::F32(out),
        )=>{

            for i in 0..input.len()
            {

                out[i]=
                    -input[i];

            }

        }




        (
            crate::tensor::Storage::F64(input),
            crate::tensor::Storage::F64(out),
        )=>{

            for i in 0..input.len()
            {

                out[i]=
                    -input[i];

            }

        }



        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "neg requires float tensor"
                    .into()
                )
            );

        }

    }



    Ok(output)

}









// =====================================================
// Scalar Operations
// =====================================================



pub fn add_scalar(
    x:&Tensor,
    value:f64,
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
            crate::tensor::Storage::F32(input),
            crate::tensor::Storage::F32(out),
        )=>{

            let v =
                value as f32;


            for i in 0..input.len()
            {

                out[i]=
                    input[i]+v;

            }

        }





        (
            crate::tensor::Storage::F64(input),
            crate::tensor::Storage::F64(out),
        )=>{


            for i in 0..input.len()
            {

                out[i]=
                    input[i]+value;

            }

        }



        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "scalar add requires float tensor"
                    .into()
                )
            );

        }

    }


    Ok(output)

}







pub fn mul_scalar(
    x:&Tensor,
    value:f64,
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
            crate::tensor::Storage::F32(input),
            crate::tensor::Storage::F32(out),
        )=>{

            let v =
                value as f32;


            for i in 0..input.len()
            {

                out[i]=
                    input[i]*v;

            }

        }




        (
            crate::tensor::Storage::F64(input),
            crate::tensor::Storage::F64(out),
        )=>{


            for i in 0..input.len()
            {

                out[i]=
                    input[i]*value;

            }

        }



        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "scalar mul requires float tensor"
                    .into()
                )
            );

        }

    }


    Ok(output)

}









// =====================================================
// Exponential
// =====================================================



pub fn exp(
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
            crate::tensor::Storage::F32(input),
            crate::tensor::Storage::F32(out),
        )=>{


            for i in 0..input.len()
            {

                out[i]=
                    input[i].exp();

            }

        }



        (
            crate::tensor::Storage::F64(input),
            crate::tensor::Storage::F64(out),
        )=>{


            for i in 0..input.len()
            {

                out[i]=
                    input[i].exp();

            }

        }



        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "exp requires float tensor"
                    .into()
                )
            );

        }

    }


    Ok(output)

}









// =====================================================
// Natural Logarithm
// =====================================================



pub fn log(
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
            crate::tensor::Storage::F32(input),
            crate::tensor::Storage::F32(out),
        )=>{


            for i in 0..input.len()
            {

                out[i]=
                    input[i].ln();

            }

        }





        (
            crate::tensor::Storage::F64(input),
            crate::tensor::Storage::F64(out),
        )=>{


            for i in 0..input.len()
            {

                out[i]=
                    input[i].ln();

            }

        }




        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "log requires float tensor"
                    .into()
                )
            );

        }

    }


    Ok(output)

}









// =====================================================
// Power
// =====================================================



pub fn pow(
    x:&Tensor,
    exponent:f64,
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
            crate::tensor::Storage::F32(input),
            crate::tensor::Storage::F32(out),
        )=>{


            let e =
                exponent as f32;


            for i in 0..input.len()
            {

                out[i]=
                    input[i].powf(e);

            }

        }





        (
            crate::tensor::Storage::F64(input),
            crate::tensor::Storage::F64(out),
        )=>{


            for i in 0..input.len()
            {

                out[i]=
                    input[i].powf(exponent);

            }

        }



        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "pow requires float tensor"
                    .into()
                )
            );

        }

    }



    Ok(output)

}


// =====================================================
// Sum Reduction
// =====================================================


pub fn sum(
    x:&Tensor,
)
    -> Result<Tensor>
{

    let mut output =
        Tensor::zeros(
            &[1],
            x.dtype(),
            x.device(),
        );



    match (
        x.storage(),
        output.storage_mut(),
    )
    {


        (
            crate::tensor::Storage::F32(input),
            crate::tensor::Storage::F32(out),
        )=>{

            out[0]=
                input
                    .iter()
                    .sum();

        }




        (
            crate::tensor::Storage::F64(input),
            crate::tensor::Storage::F64(out),
        )=>{


            out[0]=
                input
                    .iter()
                    .sum();

        }



        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "sum requires float tensor"
                    .into()
                )
            );

        }

    }



    Ok(output)

}









// =====================================================
// Mean Reduction
// =====================================================



pub fn mean(
    x:&Tensor,
)
    -> Result<Tensor>
{


    let mut output =
        sum(x)?;



    let n =
        x.numel() as f64;



    match output.storage_mut()
    {

        crate::tensor::Storage::F32(data)=>{

            data[0]/=
                n as f32;

        }



        crate::tensor::Storage::F64(data)=>{

            data[0]/=
                n;

        }



        _=>{}

    }



    Ok(output)

}









// =====================================================
// Matrix Multiplication
// =====================================================



/// Matrix multiplication.
///
/// Supports:
///
/// [m,k] * [k,n]
///
/// -> [m,n]
pub fn matmul(
    a:&Tensor,
    b:&Tensor,
)
    -> Result<Tensor>
{

    if a.ndim()!=2 ||
       b.ndim()!=2
    {

        return Err(
            TensorError::ShapeMismatch {
                expected:
                    "2D matrices".into(),

                got:
                    format!(
                        "{}D and {}D",
                        a.ndim(),
                        b.ndim()
                    ),
            }
        );

    }



    let m =
        a.shape().dims()[0];

    let k =
        a.shape().dims()[1];

    let kb =
        b.shape().dims()[0];

    let n =
        b.shape().dims()[1];



    if k != kb {

        return Err(
            TensorError::ShapeMismatch {
                expected:
                    format!(
                        "[{},k] * [k,{}]",
                        m,
                        n
                    ),

                got:
                    format!(
                        "{:?} * {:?}",
                        a.shape(),
                        b.shape()
                    ),
            }
        );

    }




    let mut output =
        Tensor::zeros(
            &[m,n],
            a.dtype(),
            a.device(),
        );





    match (
        a.storage(),
        b.storage(),
        output.storage_mut(),
    )
    {


        (
            crate::tensor::Storage::F32(x),
            crate::tensor::Storage::F32(y),
            crate::tensor::Storage::F32(out),
        )=>{


            for i in 0..m
            {

                for j in 0..n
                {

                    let mut sum =
                        0.0;


                    for p in 0..k
                    {

                        sum +=
                            x[i*k+p]
                            *
                            y[p*n+j];

                    }


                    out[i*n+j]=sum;

                }

            }


        }






        (
            crate::tensor::Storage::F64(x),
            crate::tensor::Storage::F64(y),
            crate::tensor::Storage::F64(out),
        )=>{


            for i in 0..m
            {

                for j in 0..n
                {

                    let mut sum =
                        0.0;


                    for p in 0..k
                    {

                        sum +=
                            x[i*k+p]
                            *
                            y[p*n+j];

                    }


                    out[i*n+j]=sum;

                }

            }

        }





        _=>{

            return Err(
                TensorError::UnsupportedOperation(
                    "matmul requires float tensors"
                    .into()
                )
            );

        }

    }



    Ok(output)

}









// =====================================================
// Autograd Helpers
// =====================================================



fn attach_binary_node(
    mut output:Tensor,
    a:&Tensor,
    b:&Tensor,
    op:Op,
)
    -> Tensor
{


    let node =
        Node::operation_node(
            op,
            Vec::new(),
            None,
        );



    output.set_node(
        node
    );


    output.set_requires_grad(
        true
    );


    let _ = a;
    let _ = b;


    output

}






fn attach_unary_node(
    mut output:Tensor,
    input:&Tensor,
    op:Op,
)
    -> Tensor
{


    let node =
        Node::operation_node(
            op,
            Vec::new(),
            None,
        );



    output.set_node(
        node
    );


    output.set_requires_grad(
        input.requires_grad()
    );


    output

}
