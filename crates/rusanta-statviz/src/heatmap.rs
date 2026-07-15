use rusanta_viz::{
    plot::line::LinePlot,
    style::Color,
};

use std::f64::consts::PI;

/// KDE configuration.
#[derive(Debug, Clone)]
pub struct Kde {
    pub bandwidth: f64,
    pub points: usize,
    pub color: Color,
}

impl Default for Kde {
    fn default() -> Self {
        Self {
            bandwidth: 0.3,
            points: 200,
            color: Color::BLUE,
        }
    }
}

/// Gaussian kernel function.
#[inline]
fn gaussian(u: f64) -> f64 {
    (-0.5 * u * u).exp() / (2.0 * PI).sqrt()
}

/// Compute KDE density values on a grid.
fn kde_density(values: &[f64], grid: &[f64], bw: f64) -> Vec<f64> {
    let n = values.len() as f64;
    grid.iter()
    .map(|&x| {
        values
        .iter()
        .map(|&v| gaussian((x - v) / bw))
        .sum::<f64>()
        / (n * bw)
    })
    .collect()
}

/// Create a KDE line plot from raw values.
///
/// Equivalent to seaborn / matplotlib `kdeplot`.
pub fn kde_plot(values: &[f64], cfg: Kde) -> LinePlot {
    assert!(!values.is_empty(), "kde requires at least one value");

    let min = values
    .iter()
    .cloned()
    .fold(f64::INFINITY, f64::min);
    let max = values
    .iter()
    .cloned()
    .fold(f64::NEG_INFINITY, f64::max);

    let grid: Vec<f64> = (0..cfg.points)
    .map(|i| {
        min + (max - min) * (i as f64 / (cfg.points - 1) as f64)
    })
    .collect();

    let density = kde_density(values, &grid, cfg.bandwidth);

    LinePlot::new(grid, density)
    .color(cfg.color)
    .label("kde")
}
