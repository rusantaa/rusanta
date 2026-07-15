// rusanta-ml/src/models/adaboost.rs
// AdaBoost (SAMME.R) — Binary Classification
// Weak learners: decision stumps
// Full, clean, usable implementation

use crate::dataset::Dataset;

/// Decision stump (weak learner)
#[derive(Debug, Clone)]
struct Stump {
    feature: usize,
    threshold: f64,
    left_class: f64,
    right_class: f64,
}

impl Stump {
    fn predict(&self, x: &[f64]) -> f64 {
        if x[self.feature] <= self.threshold {
            self.left_class
        } else {
            self.right_class
        }
    }
}

/// AdaBoost Classifier
#[derive(Debug, Clone)]
pub struct AdaBoostClassifier {
    pub n_estimators: usize,

    learners: Vec<(Stump, f64)>, // (weak learner, alpha)
    fitted: bool,
}

impl AdaBoostClassifier {
    pub fn new() -> Self {
        Self {
            n_estimators: 50,
            learners: Vec::new(),
            fitted: false,
        }
    }

    fn sign(v: f64) -> f64 {
        if v >= 0.0 { 1.0 } else { -1.0 }
    }
}

impl<D> AdaBoostClassifier
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit(&mut self, data: &D) {
        let n = data.len();
        let d = data.n_features();
        assert!(n > 0 && d > 0);

        // convert labels {0,1} -> {-1, +1}
        let y: Vec<f64> = (0..n)
        .map(|i| if data.target(i) > 0.0 { 1.0 } else { -1.0 })
        .collect();

        let mut weights = vec![1.0 / n as f64; n];
        self.learners.clear();

        for _ in 0..self.n_estimators {
            let mut best_err = f64::INFINITY;
            let mut best_stump = None;

            for feature in 0..d {
                let mut values: Vec<(f64, usize)> = (0..n)
                .map(|i| (data.feature_row(i)[feature], i))
                .collect();

                values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                for i in 0..n - 1 {
                    if values[i].0 == values[i + 1].0 {
                        continue;
                    }

                    let threshold = values[i].0;
                    let mut err = 0.0;

                    for j in 0..n {
                        let pred = if data.feature_row(j)[feature] <= threshold {
                            1.0
                        } else {
                            -1.0
                        };
                        if pred != y[j] {
                            err += weights[j];
                        }
                    }

                    if err < best_err {
                        best_err = err;
                        best_stump = Some(Stump {
                            feature,
                            threshold,
                            left_class: 1.0,
                            right_class: -1.0,
                        });
                    }
                }
            }

            let stump = match best_stump {
                Some(s) => s,
                None => break,
            };

            let err = best_err.clamp(1e-10, 1.0 - 1e-10);
            let alpha = 0.5 * ((1.0 - err) / err).ln();

            // update weights
            let mut norm = 0.0;
            for i in 0..n {
                let pred = stump.predict(data.feature_row(i));
                weights[i] *= (-alpha * y[i] * pred).exp();
                norm += weights[i];
            }

            for w in &mut weights {
                *w /= norm;
            }

            self.learners.push((stump, alpha));
        }

        self.fitted = true;
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        assert!(self.fitted);

        let mut score = 0.0;
        for (stump, alpha) in &self.learners {
            score += alpha * stump.predict(x);
        }

        if Self::sign(score) > 0.0 { 1.0 } else { 0.0 }
    }

    pub fn predict_batch(&self, data: &D) -> Vec<f64> {
        (0..data.len())
        .map(|i| self.predict(data.feature_row(i)))
        .collect()
    }
}
