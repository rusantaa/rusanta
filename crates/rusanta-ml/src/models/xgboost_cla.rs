// rusanta-ml/src/models/xgboost_cla.rs
// XGBoost-style Gradient Boosted Trees — Binary Classification
// Logistic loss, CART stumps, complete & usable

use crate::dataset::Dataset;

/// Decision stump (depth-1 tree)
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

/// XGBoost Binary Classifier
#[derive(Debug, Clone)]
pub struct XGBoostClassifier {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub lambda: f64, // L2 regularization
    pub gamma: f64,  // minimum split gain

    trees: Vec<Stump>,
    base_score: f64,
    fitted: bool,
}

impl XGBoostClassifier {
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

    #[inline]
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }

    fn logistic_grad(y: f64, y_hat: f64) -> (f64, f64) {
        // y ∈ {0,1}
        let p = Self::sigmoid(y_hat);
        let grad = p - y;
        let hess = p * (1.0 - p);
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

impl<D> XGBoostClassifier
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit(&mut self, data: &D) {
        let n = data.len();
        let d = data.n_features();
        assert!(n > 0 && d > 0);

        let y: Vec<f64> = (0..n).map(|i| data.target(i)).collect();

        // base score = logit(mean(y))
        let pos_rate = Self::mean(&y).clamp(1e-6, 1.0 - 1e-6);
        self.base_score = (pos_rate / (1.0 - pos_rate)).ln();

        let mut y_hat = vec![self.base_score; n];
        self.trees.clear();

        for _ in 0..self.n_estimators {
            let mut grads = vec![0.0; n];
            let mut hess = vec![0.0; n];

            for i in 0..n {
                let (g, h) = Self::logistic_grad(y[i], y_hat[i]);
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
                        best_gain = gain;
                        best_stump = Some(Stump {
                            feature,
                            threshold: values[i].0,
                            left_value: Self::calc_weight(gl, hl, self.lambda),
                                          right_value: Self::calc_weight(gr, hr, self.lambda),
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

    pub fn predict_proba(&self, x: &[f64]) -> f64 {
        assert!(self.fitted);

        let mut score = self.base_score;
        for tree in &self.trees {
            score += self.learning_rate * tree.predict(x);
        }
        Self::sigmoid(score)
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        if self.predict_proba(x) >= 0.5 {
            1.0
        } else {
            0.0
        }
    }

    pub fn predict_batch(&self, data: &D) -> Vec<f64> {
        (0..data.len())
        .map(|i| self.predict(data.feature_row(i)))
        .collect()
    }
}
