// rusanta-ml/src/models/logisticreg.rs
// Logistic Regression — binary classification, full implementation

use crate::dataset::Dataset;
use crate::pipeline::Estimator;

/// Logistic Regression (binary, sigmoid + GD)
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    pub lr: f64,
    pub epochs: usize,
    pub l2: f64,
    weights: Vec<f64>,
    bias: f64,
    fitted: bool,
}

impl LogisticRegression {
    pub fn new() -> Self {
        Self {
            lr: 0.01,
            epochs: 1000,
            l2: 0.0,
            weights: Vec::new(),
            bias: 0.0,
            fitted: false,
        }
    }

    #[inline]
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    #[inline]
    fn linear(&self, x: &[f64]) -> f64 {
        self.weights
        .iter()
        .zip(x.iter())
        .map(|(w, xi)| w * xi)
        .sum::<f64>()
        + self.bias
    }
}

impl<D> Estimator<D> for LogisticRegression
where
D: Dataset<Feature = f64, Target = f64>,
{
    fn fit(&mut self, data: &D) {
        let n = data.len();
        let d = data.n_features();

        assert!(n > 0, "empty dataset");
        assert!(d > 0, "no features");

        self.weights = vec![0.0; d];
        self.bias = 0.0;

        for _ in 0..self.epochs {
            let mut grad_w = vec![0.0; d];
            let mut grad_b = 0.0;

            for i in 0..n {
                let x = data.feature_row(i);
                let y = data.get_target(i); // expect 0.0 or 1.0

                let z = self.linear(x);
                let p = Self::sigmoid(z);
                let err = p - y;

                for j in 0..d {
                    grad_w[j] += err * x[j];
                }
                grad_b += err;
            }

            for j in 0..d {
                grad_w[j] = grad_w[j] / n as f64 + self.l2 * self.weights[j];
                self.weights[j] -= self.lr * grad_w[j];
            }
            self.bias -= self.lr * grad_b / n as f64;
        }

        self.fitted = true;
    }

    fn predict(&self, data: &D) -> Vec<f64> {
        assert!(self.fitted, "LogisticRegression not fitted");

        (0..data.len())
        .map(|i| {
            let z = self.linear(data.feature_row(i));
            if Self::sigmoid(z) >= 0.5 { 1.0 } else { 0.0 }
        })
        .collect()
    }
}
