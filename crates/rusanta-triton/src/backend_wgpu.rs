// rusanta-triton/src/backend_wgpu.rs

//! Cross-platform GPU backend using WGPU.
//!
//! This backend executes compute kernels through the WebGPU API,
//! allowing Rusanta to run on:
//! - Windows (DX12)
//! - Linux (Vulkan)
//! - macOS (Metal)
//! - WebAssembly (WebGPU)

use rusanta_core::Result;

/// Cross-platform GPU executor.
pub struct WgpuBackend {
    #[cfg(feature = "wgpu-backend")]
    device: wgpu::Device,

    #[cfg(feature = "wgpu-backend")]
    queue: wgpu::Queue,
}

impl WgpuBackend {
    /// Initialize the GPU backend.
    #[cfg(feature = "wgpu-backend")]
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("No compatible GPU adapter found.");

        let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor::default(),
                        None,
        )
        .await?;

        Ok(Self { device, queue })
    }

    /// Compile WGSL compute shader.
    #[cfg(feature = "wgpu-backend")]
    pub fn compile_kernel(
        &self,
        source: &str,
    ) -> wgpu::ShaderModule {
        self.device.create_shader_module(
            wgpu::ShaderModuleDescriptor {
                label: Some("Rusanta WGSL Kernel"),
                                         source: wgpu::ShaderSource::Wgsl(source.into()),
            },
        )
    }

    /// Execute a compute kernel.
    ///
    /// (Buffer binding and pipeline setup will be implemented
    /// in future iterations.)
    #[cfg(feature = "wgpu-backend")]
    pub fn launch(
        &self,
        shader: &wgpu::ShaderModule,
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        let _ = shader;
        let _ = workgroups;

        // TODO
        // - create compute pipeline
        // - bind storage buffers
        // - dispatch workgroups
        // - synchronize

        Ok(())
    }
}
