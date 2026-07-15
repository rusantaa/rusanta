use rusanta_viz::{
    plot::{line::LinePlot, scatter::ScatterPlot},
    style::Color,
};

/// Result of a simple linear regression.
#[derive(Debug, Clone)]
pub struct LinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub r2: f64,
}

impl LinearRegression {
    /// Predict y for a given x.
    #[inline]
    pub fn predict(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

/// Ordinary Least Squares (OLS) linear regression.
///
/// Returns slope, intercept, and R².
pub fn linear_regression(x: &[f64], y: &[f64]) -> LinearRegression {
    assert!(!x.is_empty(), "regression requires data");
    assert!(x.len() == y.len(), "x and y must have equal length");

    let n = x.len() as f64;

    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        ss_xy += (xi - mean_x) * (yi - mean_y);
        ss_xx += (xi - mean_x) * (xi - mean_x);
    }

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let y_hat = slope * xi + intercept;
        ss_tot += (yi - mean_y) * (yi - mean_y);
        ss_res += (yi - y_hat) * (yi - y_hat);
    }

    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };

    LinearRegression {
        slope,
        intercept,
        r2,
    }
}

/// Create a regression line plot from raw x/y data.
///
/// The line spans [min(x), max(x)].
pub fn regression_line(
    x: &[f64],
    y: &[f64],
    color: Color,
) -> (LinePlot, LinearRegression) {
    let model = linear_regression(x, y);

    let min_x = x
    .iter()
    .cloned()
    .fold(f64::INFINITY, f64::min);
    let max_x = x
    .iter()
    .cloned()
    .fold(f64::NEG_INFINITY, f64::max);

    let xs = vec![min_x, max_x];
    let ys = vec![
        model.predict(min_x),
        model.predict(max_x),
    ];

    let line = LinePlot::new(xs, ys)
    .color(color)
    .label("regression");

    (line, model)
}

/// Scatter + regression helper.
///
/// Equivalent to seaborn `regplot`.
pub fn regplot(
    x: &[f64],
    y: &[f64],
    scatter_color: Color,
    line_color: Color,
) -> (ScatterPlot, LinePlot, LinearRegression) {
    let scatter = ScatterPlot::new(x.to_vec(), y.to_vec())
    .color(scatter_color)
    .label("data");

    let (line, model) = regression_line(x, y, line_color);

    (scatter, line, model)
}
