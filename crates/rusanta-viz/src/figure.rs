use crate::axes::Axes;
use crate::backend::Backend;
use crate::style::Style;

/// A Figure is the top-level container for plots.
///
/// It owns one or more Axes and is responsible for rendering
/// them using a backend (SVG, PNG, GPU, etc.).
pub struct Figure {
    width: u32,
    height: u32,
    dpi: u32,
    axes: Vec<Axes>,
    style: Style,
}

impl Figure {
    /// Create a new Figure with given size (in inches) and DPI.
    pub fn new(width: f32, height: f32) -> Self {
        let dpi = 100;
        Self {
            width: (width * dpi as f32) as u32,
            height: (height * dpi as f32) as u32,
            dpi,
            axes: Vec::new(),
            style: Style::default(),
        }
    }

    /// Set DPI.
    pub fn dpi(mut self, dpi: u32) -> Self {
        self.width = self.width * dpi / self.dpi;
        self.height = self.height * dpi / self.dpi;
        self.dpi = dpi;
        self
    }

    /// Set global style.
    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    /// Add a new Axes to the figure.
    pub fn add_axes(&mut self, axes: Axes) {
        self.axes.push(axes);
    }

    /// Create and add a default Axes.
    pub fn axes(&mut self) -> &mut Axes {
        self.axes.push(Axes::new());
        self.axes.last_mut().unwrap()
    }

    /// Get immutable reference to axes.
    pub fn all_axes(&self) -> &[Axes] {
        &self.axes
    }

    /// Render the figure using a backend.
    pub fn render<B: Backend>(&self, backend: &mut B) -> crate::Result<()> {
        backend.begin(self.width, self.height, self.dpi)?;

        for axes in &self.axes {
            axes.render(backend, &self.style)?;
        }

        backend.end()
    }
}
