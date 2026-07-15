//! Rusanta Core
//!
//! Foundational primitives shared across all Rusanta crates.
//!
//! This crate intentionally contains:
//! - no domain-specific logic
//! - no heavy dependencies
//! - no allocations beyond necessity

pub mod error;
pub mod traits;
pub mod memory;
pub mod utils;

// ========================
// Re-exports (Public API)
// ========================

pub use error::{RusantaError, Result};

pub use traits::numeric::*;
pub use traits::dataset::*;
pub use traits::model::*;

pub use memory::buffer::Buffer;
pub use memory::view::{View, ViewMut};

pub use utils::math;
pub use utils::io;
