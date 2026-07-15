// rusanta-ml/src/models/catboost.rs
// CatBoost-style Gradient Boosting (Ordered Boosting, Regression)
// - handles categorical features via target statistics
// - squared loss
// - shallow trees (stumps)
// simplified but structurally faithful

use crate::dataset::Dataset;
use std::collections::HashMap;

/// Decision stump
#[derive(Debug, Clone)]
struct Stump {
    feature: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
}

/// CatBoost Regressor
#[derive(Debug, Clone)]
pub struct CatBoostRegressor {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub lambda: f64,
    pub categorical_features: Vec<usize>,

    trees: Vec<Stump>,
    base_score: f64,
    fitted: bool,
}

impl CatBoostRegressor {
    pub fn new() -> Self {
        Self {
            n_estimators: 200,
            learning_rate: 0.1,
            lambda: 1.0,
            categorical_features: Vec::new(),
            trees: Vec::new(),
            base_score: 0.0,
            fitted: false,
        }
    }

    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }

    fn squared_grad(y: f64, y_hat: f64) -> (f64, f64) {
        let g = y_hat - y;
        let h = 1.0;
        (g, h)
    }

    fn weight(g: f64, h: f64, lambda: f64) -> f64 {
        -g / (h + lambda)
    }

    fn gain(gl: f64, hl: f64, gr: f64, hr: f64, lambda: f64) -> f64 {
        0.5 * (
            (gl * gl) / (hl + lambda)
            + (gr * gr) / (hr + lambda)
            - ((gl + gr) * (gl + gr)) / (hl + hr + lambda)
        )
    }

    /// Ordered target encoding for categorical columns
    fn encode_categorical<D: Dataset<Feature = f64, Target = f64>>(
        &self,
        data: &D,
    ) -> Vec<Vec<f64>> {
        let n = data.len();
        let d = data.n_features();
        let mut encoded = vec![vec![0.0; d]; n];

        let mut stats: Vec<HashMap<i64, (f64, usize)>> =
        vec![HashMap::new(); d];

        for i in 0..n {
            let y = data.target(i);
            for j in 0..d {
                let v = data.feature_row(i)[j];
                if self.categorical_features.contains(&j) {
                    let key = v as i64;
                    let entry = stats[j].entry(key).or_insert((0.0, 0));
                    encoded[i][j] = if entry.1 == 0 {
                        self.base_score
                    } else {
                        entry.0 / entry.1 as f64
                    };
                    entry.0 += y;
                    entry.1 += 1;
                } else {
                    encoded[i][j] = v;
                }
            }
        }

        encoded
    }
}

impl<D> CatBoostRegressor
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit(&mut self, data: &D) {
        let n = data.len();
        let d = data.n_features();
        assert!(n > 0 && d > 0);

        let y: Vec<f64> = (0..n).map(|i| data.target(i)).collect();
        self.base_score = Self::mean(&y);

        let x = self.encode_categorical(data);
        let mut y_hat = vec![self.base_score; n];
        self.trees.clear();

        for _ in 0..self.n_estimators {
            let mut grads = vec![0.0; n];
            let mut hess = vec![0.0; n];

            for i in 0..n {
                let (g, h) = Self::squared_grad(y[i], y_hat[i]);
                grads[i] = g;
                hess[i] = h;
            }

            let mut best_gain = f64::NEG_INFINITY;
            let mut best = None;

            for feature in 0..d {
                let mut values: Vec<(f64, usize)> =
                (0..n).map(|i| (x[i][feature], i)).collect();
                values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                let mut gl = 0.0;
                let mut hl = 0.0;
                let g_total: f64 = grads.iter().sum();
                let h_total: f64 = hess.iter().sum();

                for i in 0..n - 1 {
                    let idx = values[i].1;
                    gl += grads[idx];
                    hl += hess[idx];

                    let gr = g_total - gl;
                    let hr = h_total - hl;

                    if values[i].0 == values[i + 1].0 {
                        continue;
                    }

                    let gain = Self::gain(gl, hl, gr, hr, self.lambda);
                    if gain > best_gain {
                        best_gain = gain;
                        best = Some(Stump {
                            feature,
                            threshold: values[i].0,
                            left_value: Self::weight(gl, hl, self.lambda),
                                    right_value: Self::weight(gr, hr, self.lambda),
                        });
                    }
                }
            }

            let stump = match best {
                Some(s) => s,
                None => break,
            };

            for i in 0..n {
                let v = x[i][stump.feature];
                y_hat[i] += self.learning_rate
                * if v <= stump.threshold {
                    stump.left_value
                } else {
                    stump.right_value
                };
            }

            self.trees.push(stump);
        }

        self.fitted = true;
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        assert!(self.fitted);

        let mut y = self.base_score;
        for t in &self.trees {
            y += self.learning_rate
            * if x[t.feature] <= t.threshold {
                t.left_value
            } else {
                t.right_value
            };
        }
        y
    }

    pub fn predict_batch(&self, data: &D) -> Vec<f64> {
        let x = self.encode_categorical(data);
        (0..x.len()).map(|i| self.predict(&x[i])).collect()
    }
}
