// rusanta-tensor/src/device/mod.rs

//! Device abstraction layer.
//!
//! A Tensor does not directly know how computation happens.
//! It only stores where its memory and execution backend live.
//!
//! Supported devices:
//!
//! - CPU
//! - GPU
//!
//! GPU execution is delegated to:
//!
//! - rusanta-triton
//!     - WGPU backend
//!     - CUDA backend


pub mod cpu;

pub mod gpu;



// =====================================================
// Device Definition
// =====================================================


/// Hardware execution device.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash
)]
pub enum Device {


    /// Host CPU execution.
    CPU,



    /// GPU execution.
    ///
    /// The value represents GPU index.
    ///
    /// Example:
    ///
    /// GPU(0)
    /// GPU(1)
    GPU(usize),

}





impl Device {


    /// Returns true if device is CPU.
    pub fn is_cpu(
        &self
    )
        -> bool
    {

        matches!(
            self,
            Device::CPU
        )

    }





    /// Returns true if device is GPU.
    pub fn is_gpu(
        &self
    )
        -> bool
    {

        matches!(
            self,
            Device::GPU(_)
        )

    }





    /// Returns GPU index.
    pub fn gpu_index(
        &self
    )
        -> Option<usize>
    {

        match self {

            Device::GPU(id)=>
                Some(*id),


            Device::CPU =>
                None,

        }

    }




    /// Default device selection.
    ///
    /// Currently CPU.
    ///
    /// Future:
    /// - detect CUDA
    /// - detect WGPU adapter
    pub fn default()
        -> Self
    {

        Device::CPU

    }


}






impl std::fmt::Display for Device {


    fn fmt(
        &self,
        f:&mut std::fmt::Formatter<'_>,
    )
        -> std::fmt::Result
    {

        match self {


            Device::CPU =>
                write!(
                    f,
                    "cpu"
                ),



            Device::GPU(index)=>
                write!(
                    f,
                    "gpu:{}",
                    index
                ),


        }

    }

}




// =====================================================
// Device Errors
// =====================================================


#[derive(
    Debug,
    Clone
)]
pub enum DeviceError {


    /// Device is unavailable.
    Unavailable(
        String
    ),



    /// Memory allocation failure.
    OutOfMemory,



    /// Unsupported operation.
    Unsupported(
        String
    ),

}



impl std::fmt::Display for DeviceError {


    fn fmt(
        &self,
        f:&mut std::fmt::Formatter<'_>,
    )
        -> std::fmt::Result
    {

        match self {


            DeviceError::Unavailable(msg)=>
                write!(
                    f,
                    "Device unavailable: {}",
                    msg
                ),



            DeviceError::OutOfMemory=>
                write!(
                    f,
                    "Device out of memory"
                ),



            DeviceError::Unsupported(msg)=>
                write!(
                    f,
                    "Unsupported device operation: {}",
                    msg
                ),


        }

    }

}



impl std::error::Error for DeviceError {}
