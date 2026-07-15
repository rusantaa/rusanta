use rusanta_viz::{
    plot::{line::LinePlot, scatter::ScatterPlot},
    style::Color,
};

use std::f64::consts::PI;

/// Violin plot configuration.
pub struct Violin {
    pub bandwidth: f64,
    pub points: usize,
    pub color: Color,
}

impl Default for Violin {
    fn default() -> Self {
        Self {
            bandwidth: 0.2,
            points: 100,
            color: Color::rgba(0.4, 0.2, 0.7, 0.6),
        }
    }
}

/// Gaussian kernel.
fn gaussian(u: f64) -> f64 {
    (-0.5 * u * u).exp() / (2.0 * PI).sqrt()
}

/// Compute KDE for a set of values.
fn kde(values: &[f64], grid: &[f64], bw: f64) -> Vec<f64> {
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

/// Create violin plot primitives at position `x`.
///
/// Returns plots to be added to an Axes.
pub fn violin_at(
    x: f64,
    values: &[f64],
    cfg: Violin,
) -> Vec<Box<dyn rusanta_viz::plot::Plot>> {
    assert!(!values.is_empty(), "violin requires data");

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

    let density = kde(values, &grid, cfg.bandwidth);
    let max_d = density
    .iter()
    .cloned()
    .fold(f64::NEG_INFINITY, f64::max);

    let scale = if max_d > 0.0 { 0.4 / max_d } else { 1.0 };

    let left_x: Vec<f64> = density
    .iter()
    .map(|d| x - d * scale)
    .collect();
    let right_x: Vec<f64> = density
    .iter()
    .map(|d| x + d * scale)
    .collect();

    let mut plots: Vec<Box<dyn rusanta_viz::plot::Plot>> = Vec::new();

    // Left side
    plots.push(Box::new(
        LinePlot::new(left_x.clone(), grid.clone()).color(cfg.color),
    ));

    // Right side
    plots.push(Box::new(
        LinePlot::new(right_x.clone(), grid.clone()).color(cfg.color),
    ));

    // Median marker
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];

    plots.push(Box::new(
        ScatterPlot::new(vec![x], vec![median])
        .size(4.0)
        .color(Color::BLACK),
    ));

    plots
}
