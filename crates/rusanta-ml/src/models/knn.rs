// rusanta-ml/src/models/linear.rs

use crate::dataset::Dataset;
use crate::pipeline::Estimator;

/// Linear regression model (Ordinary Least Squares).
///
/// This is the most fundamental supervised model:
/// - numeric features
/// - numeric target
/// - closed-form solution
#[derive(Debug, Clone)]
pub struct LinearRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
    fitted: bool,
}

impl LinearRegression {
    /// Create a new, unfitted linear regression model.
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            fitted: false,
        }
    }
}

impl<D> Estimator<D> for LinearRegression
where
D: Dataset<Feature = f64, Target = f64>,
{
    /// Fit using Ordinary Least Squares.
    fn fit(&mut self, data: &D) {
        let n = data.len();
        let d = data.n_features();

        assert!(n > 0, "dataset is empty");
        assert!(d > 0, "no features");

        // Initialize
        self.weights = vec![0.0; d];
        self.bias = 0.0;

        // Compute means
        let mut mean_x = vec![0.0; d];
        let mut mean_y = 0.0;

        for i in 0..n {
            let row = data.feature_row(i);
            for j in 0..d {
                mean_x[j] += row[j];
            }
            mean_y += data.get_target(i);
        }

        for j in 0..d {
            mean_x[j] /= n as f64;
        }
        mean_y /= n as f64;

        // Compute covariance and variance
        let mut cov_xy = vec![0.0; d];
        let mut var_x = vec![0.0; d];

        for i in 0..n {
            let row = data.feature_row(i);
            let y = data.get_target(i);

            for j in 0..d {
                cov_xy[j] += (row[j] - mean_x[j]) * (y - mean_y);
                var_x[j] += (row[j] - mean_x[j]).powi(2);
            }
        }

        for j in 0..d {
            if var_x[j] != 0.0 {
                self.weights[j] = cov_xy[j] / var_x[j];
            } else {
                self.weights[j] = 0.0;
            }
        }

        // Bias term
        self.bias = mean_y
        - self
        .weights
        .iter()
        .zip(mean_x.iter())
        .map(|(w, mx)| w * mx)
        .sum::<f64>();

        self.fitted = true;
    }

    /// Predict using the fitted model.
    fn predict(&self, data: &D) -> Vec<f64> {
        assert!(self.fitted, "model not fitted");

        let mut preds = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            let row = data.feature_row(i);
            let mut y = self.bias;

            for j in 0..row.len() {
                y += self.weights[j] * row[j];
            }

            preds.push(y);
        }

        preds
    }
}
