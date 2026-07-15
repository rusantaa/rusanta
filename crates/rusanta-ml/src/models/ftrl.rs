// rusanta-ml/src/models/ftrl.rs
// FTRL-Proximal — online logistic regression with L1/L2 regularization

use crate::dataset::Dataset;
use crate::pipeline::Estimator;

/// FTRL-Proximal optimizer / model
///
/// References:
/// - McMahan et al., "Ad Click Prediction: a View from the Trenches"
///
/// Supports:
/// - binary classification
/// - online / incremental updates
#[derive(Debug, Clone)]
pub struct FTRL {
    pub alpha: f64, // learning rate
    pub beta: f64,
    pub l1: f64,
    pub l2: f64,

    z: Vec<f64>,
    n: Vec<f64>,
    w: Vec<f64>,
    bias: f64,
    fitted: bool,
}

impl FTRL {
    pub fn new() -> Self {
        Self {
            alpha: 0.1,
            beta: 1.0,
            l1: 1.0,
            l2: 1.0,
            z: Vec::new(),
            n: Vec::new(),
            w: Vec::new(),
            bias: 0.0,
            fitted: false,
        }
    }

    #[inline]
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    fn compute_weights(&mut self) {
        for i in 0..self.z.len() {
            if self.z[i].abs() <= self.l1 {
                self.w[i] = 0.0;
            } else {
                let sign = self.z[i].signum();
                self.w[i] = -(self.z[i] - sign * self.l1)
                / ((self.beta + self.n[i].sqrt()) / self.alpha + self.l2);
            }
        }
    }

    pub fn partial_fit(&mut self, x: &[f64], y: f64) {
        if self.z.is_empty() {
            let d = x.len();
            self.z = vec![0.0; d];
            self.n = vec![0.0; d];
            self.w = vec![0.0; d];
        }

        self.compute_weights();

        let wx: f64 = self
        .w
        .iter()
        .zip(x.iter())
        .map(|(w, xi)| w * xi)
        .sum::<f64>()
        + self.bias;

        let p = Self::sigmoid(wx);
        let g = p - y;

        for i in 0..x.len() {
            let gi = g * x[i];
            let sigma = (self.n[i] + gi * gi).sqrt() - self.n[i].sqrt();
            let sigma = sigma / self.alpha;

            self.z[i] += gi - sigma * self.w[i];
            self.n[i] += gi * gi;
        }

        self.bias -= self.alpha * g;
    }
}

impl<D> Estimator<D> for FTRL
where
D: Dataset<Feature = f64, Target = f64>,
{
    fn fit(&mut self, data: &D) {
        for i in 0..data.len() {
            let x = data.feature_row(i);
            let y = data.get_target(i); // expect 0.0 or 1.0
            self.partial_fit(x, y);
        }
        self.fitted = true;
    }

    fn predict(&self, data: &D) -> Vec<f64> {
        assert!(self.fitted, "FTRL not fitted");

        (0..data.len())
        .map(|i| {
            let x = data.feature_row(i);
            let z: f64 = self
            .w
            .iter()
            .zip(x.iter())
            .map(|(w, xi)| w * xi)
            .sum::<f64>()
            + self.bias;

            if Self::sigmoid(z) >= 0.5 { 1.0 } else { 0.0 }
        })
        .collect()
    }
}
