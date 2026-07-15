use crate::backend::Backend;
use crate::plot::Plot;
use crate::style::Style;

/// Axes represent a single plotting area inside a Figure.
///
/// It owns:
/// - plots (line, scatter, bar, ...)
/// - axis limits
/// - labels and title
pub struct Axes {
    plots: Vec<Box<dyn Plot>>,
    xlim: Option<(f64, f64)>,
    ylim: Option<(f64, f64)>,
    xlabel: Option<String>,
    ylabel: Option<String>,
    title: Option<String>,
}

impl Axes {
    /// Create a new empty Axes.
    pub fn new() -> Self {
        Self {
            plots: Vec::new(),
            xlim: None,
            ylim: None,
            xlabel: None,
            ylabel: None,
            title: None,
        }
    }

    /// Add a plot to the axes.
    pub fn add_plot<P: Plot + 'static>(&mut self, plot: P) {
        self.plots.push(Box::new(plot));
    }

    /// Set x-axis limits.
    pub fn xlim(mut self, min: f64, max: f64) -> Self {
        self.xlim = Some((min, max));
        self
    }

    /// Set y-axis limits.
    pub fn ylim(mut self, min: f64, max: f64) -> Self {
        self.ylim = Some((min, max));
        self
    }

    /// Set x-axis label.
    pub fn xlabel<S: Into<String>>(mut self, label: S) -> Self {
        self.xlabel = Some(label.into());
        self
    }

    /// Set y-axis label.
    pub fn ylabel<S: Into<String>>(mut self, label: S) -> Self {
        self.ylabel = Some(label.into());
        self
    }

    /// Set title.
    pub fn title<S: Into<String>>(mut self, title: S) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Internal render entry point.
    pub(crate) fn render<B: Backend>(
        &self,
        backend: &mut B,
        style: &Style,
    ) -> crate::Result<()> {
        backend.begin_axes(
            self.xlim,
            self.ylim,
            self.xlabel.as_deref(),
                           self.ylabel.as_deref(),
                           self.title.as_deref(),
                           style,
        )?;

        for plot in &self.plots {
            plot.render(backend, style)?;
        }

        backend.end_axes()
    }
}

impl Default for Axes {
    fn default() -> Self {
        Self::new()
    }
}
