// rusanta-ml/src/models/gmm.rs
// Gaussian Mixture Model — EM algorithm (full, numeric, diagonal covariance)

use crate::dataset::Dataset;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::f64::consts::PI;

/// Gaussian Mixture Model
#[derive(Debug, Clone)]
pub struct GaussianMixture {
    pub k: usize,
    pub max_iters: usize,
    pub tol: f64,

    weights: Vec<f64>,          // mixing coefficients
    means: Vec<Vec<f64>>,       // component means
    vars: Vec<Vec<f64>>,        // diagonal variances
    fitted: bool,
}

impl GaussianMixture {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iters: 100,
            tol: 1e-4,
            weights: Vec::new(),
            means: Vec::new(),
            vars: Vec::new(),
            fitted: false,
        }
    }

    fn gaussian_log_pdf(x: &[f64], mean: &[f64], var: &[f64]) -> f64 {
        let d = x.len();
        let mut logp = 0.0;

        for j in 0..d {
            let v = var[j].max(1e-9);
            let diff = x[j] - mean[j];
            logp += -0.5 * ((diff * diff) / v + v.ln() + (2.0 * PI).ln());
        }
        logp
    }
}

impl<D> GaussianMixture
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit(&mut self, data: &D) {
        let n = data.len();
        let d = data.n_features();
        assert!(n > 0 && d > 0 && self.k > 0);

        let mut rng = thread_rng();

        // init weights
        self.weights = vec![1.0 / self.k as f64; self.k];

        // init means from random samples
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(&mut rng);
        self.means = idx[..self.k]
        .iter()
        .map(|&i| data.feature_row(i).to_vec())
        .collect();

        // init variances
        self.vars = vec![vec![1.0; d]; self.k];

        let mut resp = vec![vec![0.0; self.k]; n];

        for _ in 0..self.max_iters {
            // ===== E-step =====
            for i in 0..n {
                let x = data.feature_row(i);

                let mut log_sum = f64::NEG_INFINITY;
                for k in 0..self.k {
                    let lp = self.weights[k].ln()
                    + Self::gaussian_log_pdf(x, &self.means[k], &self.vars[k]);
                    resp[i][k] = lp;
                    log_sum = log_sum.max(lp);
                }

                // log-sum-exp
                let mut sum = 0.0;
                for k in 0..self.k {
                    resp[i][k] = (resp[i][k] - log_sum).exp();
                    sum += resp[i][k];
                }
                for k in 0..self.k {
                    resp[i][k] /= sum;
                }
            }

            // ===== M-step =====
            let mut max_shift = 0.0;

            for k in 0..self.k {
                let nk: f64 = resp.iter().map(|r| r[k]).sum();

                // update weight
                self.weights[k] = nk / n as f64;

                // update mean
                let mut new_mean = vec![0.0; d];
                for i in 0..n {
                    let x = data.feature_row(i);
                    for j in 0..d {
                        new_mean[j] += resp[i][k] * x[j];
                    }
                }
                for j in 0..d {
                    new_mean[j] /= nk;
                }

                // update variance
                let mut new_var = vec![0.0; d];
                for i in 0..n {
                    let x = data.feature_row(i);
                    for j in 0..d {
                        let diff = x[j] - new_mean[j];
                        new_var[j] += resp[i][k] * diff * diff;
                    }
                }
                for j in 0..d {
                    new_var[j] = (new_var[j] / nk).max(1e-9);
                }

                // convergence check
                for j in 0..d {
                    max_shift = max_shift.max((self.means[k][j] - new_mean[j]).abs());
                }

                self.means[k] = new_mean;
                self.vars[k] = new_var;
            }

            if max_shift < self.tol {
                break;
            }
        }

        self.fitted = true;
    }

    pub fn predict(&self, data: &D) -> Vec<usize> {
        assert!(self.fitted, "GaussianMixture not fitted");

        (0..data.len())
        .map(|i| {
            let x = data.feature_row(i);

            (0..self.k)
            .map(|k| {
                self.weights[k].ln()
                + Self::gaussian_log_pdf(x, &self.means[k], &self.vars[k])
            })
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
        })
        .collect()
    }
}
