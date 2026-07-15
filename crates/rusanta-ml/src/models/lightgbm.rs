// rusanta-ml/src/models/lightgbm.rs
// LightGBM-style Gradient Boosted Trees (Histogram-based, Regression)
// - squared loss
// - histogram binning
// - decision stumps (depth-1)
// clean + complete + usable

use crate::dataset::Dataset;

/// Histogram-based decision stump
#[derive(Debug, Clone)]
struct HistStump {
    feature: usize,
    bin_threshold: usize,
    left_value: f64,
    right_value: f64,
}

/// LightGBM Regressor
#[derive(Debug, Clone)]
pub struct LightGBMRegressor {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub lambda: f64,
    pub gamma: f64,
    pub max_bins: usize,

    trees: Vec<HistStump>,
    bin_edges: Vec<Vec<f64>>,
    base_score: f64,
    fitted: bool,
}

impl LightGBMRegressor {
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            lambda: 1.0,
            gamma: 0.0,
            max_bins: 255,
            trees: Vec::new(),
            bin_edges: Vec::new(),
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

    fn calc_weight(g: f64, h: f64, lambda: f64) -> f64 {
        -g / (h + lambda)
    }

    fn gain(gl: f64, hl: f64, gr: f64, hr: f64, lambda: f64, gamma: f64) -> f64 {
        0.5 * (
            (gl * gl) / (hl + lambda)
            + (gr * gr) / (hr + lambda)
            - ((gl + gr) * (gl + gr)) / (hl + hr + lambda)
        ) - gamma
    }

    fn build_bins<D: Dataset<Feature = f64, Target = f64>>(
        &self,
        data: &D,
    ) -> Vec<Vec<f64>> {
        let d = data.n_features();
        let n = data.len();
        let mut bins = Vec::with_capacity(d);

        for j in 0..d {
            let mut col: Vec<f64> = (0..n)
            .map(|i| data.feature_row(i)[j])
            .collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let step = (n / self.max_bins).max(1);
            let edges: Vec<f64> = col.iter().step_by(step).cloned().collect();
            bins.push(edges);
        }
        bins
    }

    fn bin_index(edges: &[f64], v: f64) -> usize {
        match edges.binary_search_by(|x| x.partial_cmp(&v).unwrap()) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        }
    }
}

impl<D> LightGBMRegressor
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit(&mut self, data: &D) {
        let n = data.len();
        let d = data.n_features();
        assert!(n > 0 && d > 0);

        let y: Vec<f64> = (0..n).map(|i| data.target(i)).collect();
        self.base_score = Self::mean(&y);
        self.bin_edges = self.build_bins(data);

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
                let bins = &self.bin_edges[feature];
                let nb = bins.len();

                let mut gl = vec![0.0; nb];
                let mut hl = vec![0.0; nb];

                for i in 0..n {
                    let b = Self::bin_index(bins, data.feature_row(i)[feature]);
                    gl[b] += grads[i];
                    hl[b] += hess[i];
                }

                let mut g_left = 0.0;
                let mut h_left = 0.0;
                let g_total: f64 = grads.iter().sum();
                let h_total: f64 = hess.iter().sum();

                for b in 0..nb - 1 {
                    g_left += gl[b];
                    h_left += hl[b];

                    let g_right = g_total - g_left;
                    let h_right = h_total - h_left;

                    let gain = Self::gain(
                        g_left,
                        h_left,
                        g_right,
                        h_right,
                        self.lambda,
                        self.gamma,
                    );

                    if gain > best_gain {
                        best_gain = gain;
                        best = Some(HistStump {
                            feature,
                            bin_threshold: b,
                            left_value: Self::calc_weight(g_left, h_left, self.lambda),
                                    right_value: Self::calc_weight(g_right, h_right, self.lambda),
                        });
                    }
                }
            }

            let stump = match best {
                Some(s) => s,
                None => break,
            };

            for i in 0..n {
                let v = data.feature_row(i)[stump.feature];
                let b = Self::bin_index(&self.bin_edges[stump.feature], v);
                y_hat[i] += self.learning_rate
                * if b <= stump.bin_threshold {
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
        for stump in &self.trees {
            let b = Self::bin_index(&self.bin_edges[stump.feature], x[stump.feature]);
            y += self.learning_rate
            * if b <= stump.bin_threshold {
                stump.left_value
            } else {
                stump.right_value
            };
        }
        y
    }

    pub fn predict_batch(&self, data: &D) -> Vec<f64> {
        (0..data.len())
        .map(|i| self.predict(data.feature_row(i)))
        .collect()
    }
}
