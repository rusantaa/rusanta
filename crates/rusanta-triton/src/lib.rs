// rusanta-triton/src/lib.rs

//! Rusanta Triton
//!
//! Native GPU compiler and execution layer for the Rusanta ecosystem.
//!
//! Supported backends:
//! - WGPU (cross-platform)
//! - CUDA (NVIDIA)
//!
//! Future:
//! - HIP (AMD ROCm)
//! - Metal native
//! - Vulkan compute
//! - OpenCL

#![forbid(unsafe_op_in_unsafe_fn)]

use rusanta_core::Result;

#[cfg(feature = "cuda-backend")]
pub mod backend_cuda;

#[cfg(feature = "wgpu-backend")]
pub mod backend_wgpu;

/// Available GPU backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Wgpu,
    Cuda,
}

/// Common GPU backend interface.
pub trait GpuBackend {
    /// Human-readable backend name.
    fn name(&self) -> &'static str;

    /// Compile a compute kernel.
    fn compile(&mut self, source: &str) -> Result<()>;

    /// Launch a compiled kernel.
    fn launch(
        &mut self,
        kernel: &str,
        workgroups: (u32, u32, u32),
    ) -> Result<()>;

    /// Wait until all queued GPU work finishes.
    fn synchronize(&mut self) -> Result<()>;
}

/// Returns the compiled default backend.
pub fn default_backend() -> BackendKind {
    #[cfg(feature = "cuda-backend")]
    {
        BackendKind::Cuda
    }

    #[cfg(all(
    not(feature = "cuda-backend"),
              feature = "wgpu-backend"
    ))]
    {
        BackendKind::Wgpu
    }
}

/// Whether CUDA support is compiled in.
pub fn cuda_available() -> bool {
    cfg!(feature = "cuda-backend")
}

/// Whether WGPU support is compiled in.
pub fn wgpu_available() -> bool {
    cfg!(feature = "wgpu-backend")
}

/// Backend information.
#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub backend: BackendKind,
    pub name: &'static str,
    pub vendor: &'static str,
    pub portable: bool,
}

impl BackendInfo {
    pub fn current() -> Self {
        match default_backend() {
            BackendKind::Cuda => Self {
                backend: BackendKind::Cuda,
                name: "CUDA",
                vendor: "NVIDIA",
                portable: false,
            },
            BackendKind::Wgpu => Self {
                backend: BackendKind::Wgpu,
                name: "WGPU",
                vendor: "Cross-platform",
                portable: true,
            },
        }
    }
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendKind::Cuda => write!(f, "CUDA"),
            BackendKind::Wgpu => write!(f, "WGPU"),
        }
    }
}

/// Returns backend information for the compiled backend.
pub fn backend_info() -> BackendInfo {
    BackendInfo::current()
}
