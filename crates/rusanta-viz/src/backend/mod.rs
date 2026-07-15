use crate::style::Color;

/// Backend trait.
///
/// This trait defines the low-level drawing primitives used by plots.
/// All coordinates are expected to be **normalized**:
///
/// - x ∈ [0, 1]
/// - y ∈ [0, 1]
///
/// Higher-level constructs (axes, scaling, layout) are handled elsewhere.
pub trait Backend {
    /// Draw a connected line plot.
    fn draw_line(
        &mut self,
        x: &[f64],
        y: &[f64],
        width: f32,
        color: Color,
        label: Option<&str>,
    ) -> crate::Result<()>;

    /// Draw a scatter plot.
    fn draw_scatter(
        &mut self,
        x: &[f64],
        y: &[f64],
        size: f32,
        color: Color,
        label: Option<&str>,
    ) -> crate::Result<()>;

    /// Draw bars (used by bar plots and histograms).
    ///
    /// `x` represents bar centers.
    /// `heights` represent bar heights (normalized).
    /// `width` is normalized relative to x spacing.
    fn draw_bars(
        &mut self,
        x: &[f64],
        heights: &[f64],
        width: f64,
        color: Color,
        label: Option<&str>,
    ) -> crate::Result<()>;
}

pub mod svg;
pub mod png;
pub mod wgpu;
