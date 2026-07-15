use crate::backend::Backend;
use crate::plot::Plot;
use crate::style::{Color, Style};

/// Histogram normalization mode.
#[derive(Debug, Clone, Copy)]
pub enum HistNorm {
    Count,
    Density,
}

/// A histogram plot.
///
/// Data is binned internally before rendering.
pub struct Histogram {
    data: Vec<f64>,
    bins: usize,
    range: Option<(f64, f64)>,
    norm: HistNorm,
    color: Option<Color>,
    label: Option<String>,
}

impl Histogram {
    /// Create a new histogram with default bins (10).
    pub fn new<D>(data: D) -> Self
    where
    D: Into<Vec<f64>>,
    {
        Self {
            data: data.into(),
            bins: 10,
            range: None,
            norm: HistNorm::Count,
            color: None,
            label: None,
        }
    }

    /// Set number of bins.
    pub fn bins(mut self, bins: usize) -> Self {
        assert!(bins > 0, "bins must be > 0");
        self.bins = bins;
        self
    }

    /// Set data range.
    pub fn range(mut self, min: f64, max: f64) -> Self {
        assert!(min < max, "invalid histogram range");
        self.range = Some((min, max));
        self
    }

    /// Normalize histogram.
    pub fn norm(mut self, norm: HistNorm) -> Self {
        self.norm = norm;
        self
    }

    /// Set bar color.
    pub fn color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    /// Set label (for legend).
    pub fn label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    fn compute_bins(&self) -> (Vec<f64>, Vec<f64>) {
        let (min, max) = self.range.unwrap_or_else(|| {
            let mut min = self.data[0];
            let mut max = self.data[0];
            for &v in &self.data {
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
            (min, max)
        });

        let width = (max - min) / self.bins as f64;
        let mut counts = vec![0.0; self.bins];

        for &v in &self.data {
            if v < min || v > max {
                continue;
            }
            let mut idx = ((v - min) / width) as usize;
            if idx == self.bins {
                idx -= 1;
            }
            counts[idx] += 1.0;
        }

        if matches!(self.norm, HistNorm::Density) {
            let total: f64 = counts.iter().sum();
            if total > 0.0 {
                for c in counts.iter_mut() {
                    *c /= total * width;
                }
            }
        }

        let centers: Vec<f64> = (0..self.bins)
        .map(|i| min + (i as f64 + 0.5) * width)
        .collect();

        (centers, counts)
    }
}

impl Plot for Histogram {
    fn render<B: Backend>(
        &self,
        backend: &mut B,
        style: &Style,
    ) -> crate::Result<()> {
        let (x, heights) = self.compute_bins();
        let color = self.color.unwrap_or(style.hist.color);
        let width = (x[1] - x[0]) * 0.9;

        backend.draw_bars(
            &x,
            &heights,
            width,
            color,
            self.label.as_deref(),
        )
    }
}
