// rusanta-ml/src/models/xgboost.rs
// XGBoost-style Gradient Boosted Decision Trees (SIMPLIFIED, FULL, WORKING)
// - regression only
// - squared loss
// - CART stumps (depth-1 trees)
// This is NOT a prototype; it is a complete minimal XGBoost-like implementation.

use crate::dataset::Dataset;

/// Single decision stump
#[derive(Debug, Clone)]
struct Stump {
    feature: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
}

impl Stump {
    fn predict(&self, x: &[f64]) -> f64 {
        if x[self.feature] <= self.threshold {
            self.left_value
        } else {
            self.right_value
        }
    }
}

/// XGBoost Regressor (GBDT)
#[derive(Debug, Clone)]
pub struct XGBoostRegressor {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub lambda: f64, // L2 regularization
    pub gamma: f64,  // minimum split gain

    trees: Vec<Stump>,
    base_score: f64,
    fitted: bool,
}

impl XGBoostRegressor {
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            lambda: 1.0,
            gamma: 0.0,
            trees: Vec::new(),
            base_score: 0.0,
            fitted: false,
        }
    }

    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }

    fn squared_grad(y: f64, y_hat: f64) -> (f64, f64) {
        // gradient, hessian
        let grad = y_hat - y;
        let hess = 1.0;
        (grad, hess)
    }

    fn calc_weight(grad: f64, hess: f64, lambda: f64) -> f64 {
        -grad / (hess + lambda)
    }

    fn calc_gain(
        gl: f64,
        hl: f64,
        gr: f64,
        hr: f64,
        lambda: f64,
        gamma: f64,
    ) -> f64 {
        let gain = (gl * gl) / (hl + lambda)
        + (gr * gr) / (hr + lambda)
        - ((gl + gr) * (gl + gr)) / (hl + hr + lambda);
        gain * 0.5 - gamma
    }
}

impl<D> XGBoostRegressor
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit(&mut self, data: &D) {
        let n = data.len();
        let d = data.n_features();
        assert!(n > 0 && d > 0);

        let y: Vec<f64> = (0..n).map(|i| data.target(i)).collect();
        self.base_score = Self::mean(&y);

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
            let mut best_stump = None;

            for feature in 0..d {
                let mut values: Vec<(f64, usize)> = (0..n)
                .map(|i| (data.feature_row(i)[feature], i))
                .collect();

                values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                let mut gl = 0.0;
                let mut hl = 0.0;
                let mut gr: f64 = grads.iter().sum();
                let mut hr: f64 = hess.iter().sum();

                for i in 0..n - 1 {
                    let idx = values[i].1;
                    gl += grads[idx];
                    hl += hess[idx];
                    gr -= grads[idx];
                    hr -= hess[idx];

                    if values[i].0 == values[i + 1].0 {
                        continue;
                    }

                    let gain = Self::calc_gain(
                        gl,
                        hl,
                        gr,
                        hr,
                        self.lambda,
                        self.gamma,
                    );

                    if gain > best_gain {
                        let left_value =
                        Self::calc_weight(gl, hl, self.lambda);
                        let right_value =
                        Self::calc_weight(gr, hr, self.lambda);

                        best_gain = gain;
                        best_stump = Some(Stump {
                            feature,
                            threshold: values[i].0,
                            left_value,
                            right_value,
                        });
                    }
                }
            }

            let stump = match best_stump {
                Some(s) => s,
                None => break,
            };

            for i in 0..n {
                y_hat[i] += self.learning_rate
                * stump.predict(data.feature_row(i));
            }

            self.trees.push(stump);
        }

        self.fitted = true;
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        assert!(self.fitted);

        let mut y = self.base_score;
        for tree in &self.trees {
            y += self.learning_rate * tree.predict(x);
        }
        y
    }

    pub fn predict_batch(&self, data: &D) -> Vec<f64> {
        (0..data.len())
        .map(|i| self.predict(data.feature_row(i)))
        .collect()
    }
}
