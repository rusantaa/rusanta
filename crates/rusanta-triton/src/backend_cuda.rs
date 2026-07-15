// rusanta-triton/src/backend_cuda.rs

//! NVIDIA CUDA backend.
//!
//! This backend uses CUDA/PTX through `cudarc` for maximum performance on
//! NVIDIA GPUs. Unlike the WGPU backend, it has direct access to CUDA
//! features such as:
//!
//! - CUDA kernels (.ptx)
//! - Shared memory
//! - Warp intrinsics
//! - Tensor Cores (future)
//! - Streams
//! - Events
//! - CUDA Graphs (future)

use rusanta_core::Result;

#[cfg(feature = "cuda-backend")]
use cudarc::{
    driver::{
        CudaContext,
        CudaFunction,
        CudaModule,
        LaunchConfig,
    },
    nvrtc::compile_ptx,
};

/// CUDA GPU executor.
pub struct CudaBackend {
    #[cfg(feature = "cuda-backend")]
    context: CudaContext,
}

impl CudaBackend {
    /// Initialize CUDA.
    #[cfg(feature = "cuda-backend")]
    pub fn new(device_index: usize) -> Result<Self> {
        let context = CudaContext::new(device_index)?;

        Ok(Self { context })
    }

    /// Compile CUDA source into PTX using NVRTC.
    #[cfg(feature = "cuda-backend")]
    pub fn compile_kernel(
        &self,
        cuda_source: &str,
    ) -> Result<CudaModule> {
        let ptx = compile_ptx(cuda_source)?;

        let module = self.context.load_module(ptx)?;

        Ok(module)
    }

    /// Retrieve a kernel function from a compiled module.
    #[cfg(feature = "cuda-backend")]
    pub fn kernel(
        &self,
        module: &CudaModule,
        name: &str,
    ) -> Result<CudaFunction> {
        Ok(module.load_function(name)?)
    }

    /// Launch a CUDA kernel.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - kernel arguments are valid
    /// - device pointers are valid
    /// - launch dimensions are correct
    #[cfg(feature = "cuda-backend")]
    pub unsafe fn launch<Args>(
        &self,
        function: &CudaFunction,
        grid: (u32, u32, u32),
                               block: (u32, u32, u32),
                               shared_mem_bytes: u32,
                               args: Args,
    ) -> Result<()>
    where
    Args: cudarc::driver::LaunchAsync,
    {
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes,
        };

        function.launch(cfg, args)?;

        Ok(())
    }

    /// Synchronize the device.
    #[cfg(feature = "cuda-backend")]
    pub fn synchronize(&self) -> Result<()> {
        self.context.synchronize()?;
        Ok(())
    }

    /// Returns the CUDA device name.
    #[cfg(feature = "cuda-backend")]
    pub fn device_name(&self) -> Result<String> {
        Ok(self.context.device().name()?)
    }

    /// Returns the compute capability (major, minor).
    #[cfg(feature = "cuda-backend")]
    pub fn compute_capability(&self) -> Result<(u16, u16)> {
        let device = self.context.device();

        Ok((
            device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? as u16,
            device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)? as u16,
        ))
    }
}
