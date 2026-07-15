//! stat-viz
//!
//! Statistical visualization utilities built on top of `rusanta-viz`.
//!
//! This crate provides higher-level plots and helpers commonly used in
//! data analysis, statistics, and exploratory data analysis (EDA),
//! similar in spirit to seaborn or statsmodels.
//!
//! ## Included modules
//!
//! - Categorical plots (box, violin)
//! - Distribution plots (histogram, KDE, ECDF)
//! - Regression plots
//! - Heatmaps
//! - Statistical themes

pub mod categorical;
pub mod distribution;
pub mod heatmap;
pub mod kde;
pub mod regression;
pub mod theme;

// Re-exports for convenience
pub use categorical::*;
pub use distribution::*;
pub use heatmap::*;
pub use kde::*;
pub use regression::*;
pub use theme::*;
