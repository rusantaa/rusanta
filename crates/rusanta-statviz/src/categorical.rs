//! Rusanta Frame
//!
//! A dataframe engine inspired by pandas and Polars,
//! designed for tight integration with the Rusanta ecosystem.
//!
//! This crate provides:
//! - DataFrame & Series
//! - IO (CSV, JSON, Parquet, XLSX, ...)
//! - GroupBy, Join, Aggregate
//! - Pivot tables
//! - Statistics & metrics
//!
//! Visualization and ML live in separate crates.

pub mod error;

pub mod series;
pub mod dataframe;
pub mod index;
pub mod stats;

pub mod io;
pub mod ops;
pub mod pivot;

pub mod dataloader;
pub mod transform;
pub mod pipeline;
pub mod trainer;
pub mod metrics;

// ---------- Re-exports (Public API) ---------- //

pub use dataframe::DataFrame;
pub use series::Series;
pub use index::Index;

pub use stats::{AggFunc, Describe};

pub use pivot::dynamic::PivotTable;

pub use error::{Error, Result};

// ---------- Prelude ---------- //

/// Common imports for Rusanta Frame users.
///
/// ```rust
/// use rusanta_frame::prelude::*;
/// ```
pub mod prelude {
    pub use crate::dataframe::DataFrame;
    pub use crate::series::Series;
    pub use crate::index::Index;

    pub use crate::stats::{AggFunc, Describe};
    pub use crate::pivot::dynamic::PivotTable;

    pub use crate::error::{Error, Result};
}
