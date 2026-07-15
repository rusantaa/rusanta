// rusanta-ml/src/models/svm.rs

use crate::dataset::Dataset;
use crate::pipeline::Estimator;

/// Kernel type (v1 only supports Linear).
#[derive(Debug, Clone)]
pub enum Kernel {
    Linear,
}

/// Support Vector Machine (SVM) using hinge loss + SGD.
///
/// Scope:
/// - binary classification
/// - numeric features (f64)
/// - targets must be {-1.0, +1.0}
///
/// This is a **correct, minimal** implementation.
/// Kernel tricks, SMO, and multi-class come later.
#[derive(Debug, Clone)]
pub struct SVM {
    pub lr: f64,
    pub lambda: f64,
    pub epochs: usize,
    pub kernel: Kernel,
    weights: Vec<f64>,
    bias: f64,
    fitted: bool,
}

impl SVM {
    /// Create a new linear SVM.
    pub fn new() -> Self {
        Self {
            lr: 0.01,
            lambda: 0.01,
            epochs: 1000,
            kernel: Kernel::Linear,
            weights: Vec::new(),
            bias: 0.0,
            fitted: false,
        }
    }

    #[inline]
    fn decision(&self, x: &[f64]) -> f64 {
        self.weights
        .iter()
        .zip(x.iter())
        .map(|(w, xi)| w * xi)
        .sum::<f64>()
        + self.bias
    }
}

impl<D> Estimator<D> for SVM
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
            for i in 0..n {
                let x = data.feature_row(i);
                let y = data.get_target(i);

                // Expect labels in {-1, +1}
                assert!(
                    y == 1.0 || y == -1.0,
                    "SVM target must be -1 or +1"
                );

                let margin = y * self.decision(x);

                if margin < 1.0 {
                    // gradient from hinge loss
                    for j in 0..d {
                        self.weights[j] +=
                        self.lr * (y * x[j] - self.lambda * self.weights[j]);
                    }
                    self.bias += self.lr * y;
                } else {
                    // regularization only
                    for j in 0..d {
                        self.weights[j] -= self.lr * self.lambda * self.weights[j];
                    }
                }
            }
        }

        self.fitted = true;
    }

    fn predict(&self, data: &D) -> Vec<f64> {
        assert!(self.fitted, "model not fitted");

        (0..data.len())
        .map(|i| {
            let score = self.decision(data.feature_row(i));
            if score >= 0.0 { 1.0 } else { -1.0 }
        })
        .collect()
    }
}
