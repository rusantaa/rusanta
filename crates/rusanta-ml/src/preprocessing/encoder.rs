// rusanta-ml/src/preprocessing/scaler.rs

use crate::dataset::{Dataset, InMemoryDataset};

/// Standardization strategy.
#[derive(Debug, Clone)]
pub enum Scaling {
    /// (x - mean) / std
    Standard,
    /// (x - min) / (max - min)
    MinMax,
}

/// Feature scaler.
///
/// Fits statistics on training data and applies scaling deterministically.
/// Equivalent to sklearn's StandardScaler / MinMaxScaler.
#[derive(Debug, Clone)]
pub struct Scaler {
    strategy: Scaling,
    mean: Vec<f64>,
    std: Vec<f64>,
    min: Vec<f64>,
    max: Vec<f64>,
    fitted: bool,
}

impl Scaler {
    /// Create a new scaler.
    pub fn new(strategy: Scaling) -> Self {
        Self {
            strategy,
            mean: Vec::new(),
            std: Vec::new(),
            min: Vec::new(),
            max: Vec::new(),
            fitted: false,
        }
    }

    /// Fit scaler statistics from dataset.
    pub fn fit<D>(&mut self, data: &D)
    where
    D: Dataset<Feature = f64>,
    {
        let n = data.len();
        let d = data.n_features();

        self.mean = vec![0.0; d];
        self.std = vec![0.0; d];
        self.min = vec![f64::INFINITY; d];
        self.max = vec![f64::NEG_INFINITY; d];

        for i in 0..n {
            let row = data.feature_row(i);
            for j in 0..d {
                let v = row[j];
                self.mean[j] += v;
                if v < self.min[j] {
                    self.min[j] = v;
                }
                if v > self.max[j] {
                    self.max[j] = v;
                }
            }
        }

        for j in 0..d {
            self.mean[j] /= n as f64;
        }

        for i in 0..n {
            let row = data.feature_row(i);
            for j in 0..d {
                let diff = row[j] - self.mean[j];
                self.std[j] += diff * diff;
            }
        }

        for j in 0..d {
            self.std[j] = (self.std[j] / n as f64).sqrt();
            if self.std[j] == 0.0 {
                self.std[j] = 1.0;
            }
            if self.max[j] == self.min[j] {
                self.max[j] = self.min[j] + 1.0;
            }
        }

        self.fitted = true;
    }

    /// Transform dataset using fitted statistics.
    pub fn transform<D>(&self, data: &D) -> InMemoryDataset<f64, D::Target>
    where
    D: Dataset<Feature = f64>,
    {
        assert!(self.fitted, "Scaler must be fitted before transform");

        let mut features = Vec::with_capacity(data.len());
        let mut targets = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            let row = data.feature_row(i);
            let mut scaled = Vec::with_capacity(row.len());

            for j in 0..row.len() {
                let v = match self.strategy {
                    Scaling::Standard => (row[j] - self.mean[j]) / self.std[j],
                    Scaling::MinMax => {
                        (row[j] - self.min[j]) / (self.max[j] - self.min[j])
                    }
                };
                scaled.push(v);
            }

            features.push(scaled);
            targets.push(data.get_target(i));
        }

        InMemoryDataset::new(features, targets)
    }
}
