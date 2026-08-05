// rusanta-tensor/src/device/cpu.rs

//! CPU execution backend.
//!
//! Provides low-level tensor operations running on the host CPU.
//!
//! Future optimizations:
//!
//! - SIMD acceleration
//! - Rayon parallel execution
//! - BLAS integration
//! - Cache-aware kernels


use crate::{
    Tensor,
    Storage,
    DType,
    Shape,
    TensorError,
    Result,
};





// =====================================================
// CPU Backend
// =====================================================


/// CPU execution engine.
#[derive(Debug, Clone, Copy)]
pub struct CpuBackend;




impl CpuBackend {


    /// Create CPU backend.
    pub fn new()
        -> Self
    {
        Self
    }






    // -------------------------------------------------
    // Allocation
    // -------------------------------------------------



    pub fn zeros(
        &self,
        shape:&[usize],
        dtype:DType,
    )
        -> Tensor
    {

        Tensor::zeros(
            shape,
            dtype,
            crate::Device::CPU
        )

    }




    pub fn ones(
        &self,
        shape:&[usize],
        dtype:DType,
    )
        -> Tensor
    {

        Tensor::ones(
            shape,
            dtype,
            crate::Device::CPU
        )

    }







    // -------------------------------------------------
    // Elementwise Operations
    // -------------------------------------------------



    pub fn add(
        &self,
        a:&Tensor,
        b:&Tensor,
    )
        -> Result<Tensor>
    {


        Tensor::same_shape(
            a,
            b
        );


        a.assert_float();


        let mut output =
            Tensor::empty_like(a);



        match (
            a.storage(),
            b.storage(),
            output.storage_mut(),
        ){


            (
                Storage::F32(x),
                Storage::F32(y),
                Storage::F32(out)
            )=>{

                for i in 0..x.len(){

                    out[i]=
                        x[i]+y[i];

                }

            }



            (
                Storage::F64(x),
                Storage::F64(y),
                Storage::F64(out)
            )=>{

                for i in 0..x.len(){

                    out[i]=
                        x[i]+y[i];

                }

            }



            _=>{

                return Err(
                    TensorError::UnsupportedOperation(
                        "add dtype mismatch".into()
                    )
                );

            }

        }


        Ok(output)

    }








    pub fn sub(
        &self,
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
        ){


            (
                Storage::F32(x),
                Storage::F32(y),
                Storage::F32(out)
            )=>{


                for i in 0..x.len(){

                    out[i]=
                        x[i]-y[i];

                }

            }



            (
                Storage::F64(x),
                Storage::F64(y),
                Storage::F64(out)
            )=>{


                for i in 0..x.len(){

                    out[i]=
                        x[i]-y[i];

                }

            }



            _=>{

                return Err(
                    TensorError::UnsupportedOperation(
                        "sub dtype mismatch".into()
                    )
                );

            }


        }


        Ok(output)

    }








    pub fn mul(
        &self,
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
        ){


            (
                Storage::F32(x),
                Storage::F32(y),
                Storage::F32(out)
            )=>{


                for i in 0..x.len(){

                    out[i]=
                        x[i]*y[i];

                }

            }



            (
                Storage::F64(x),
                Storage::F64(y),
                Storage::F64(out)
            )=>{


                for i in 0..x.len(){

                    out[i]=
                        x[i]*y[i];

                }

            }



            _=>{

                return Err(
                    TensorError::UnsupportedOperation(
                        "mul dtype mismatch".into()
                    )
                );

            }


        }


        Ok(output)

    }









    // -------------------------------------------------
    // Reduction Operations
    // -------------------------------------------------



    pub fn sum(
        &self,
        tensor:&Tensor,
    )
        -> Result<Tensor>
    {


        let mut output =
            Tensor::zeros(
                &[1],
                tensor.dtype(),
                crate::Device::CPU
            );



        match (
            tensor.storage(),
            output.storage_mut(),
        ){


            (
                Storage::F32(data),
                Storage::F32(out)
            )=>{

                out[0]=
                    data.iter().sum();

            }



            (
                Storage::F64(data),
                Storage::F64(out)
            )=>{

                out[0]=
                    data.iter().sum();

            }



            _=>{

                return Err(
                    TensorError::UnsupportedOperation(
                        "sum requires float tensor".into()
                    )
                );

            }


        }


        Ok(output)

    }






}
