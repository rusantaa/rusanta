// rusanta-ml/src/models/ica.rs
// Independent Component Analysis (FastICA)
// Numeric data, batch mode, whitening + fixed-point iteration

use crate::dataset::Dataset;
use rand::thread_rng;
use rand::seq::SliceRandom;

/// FastICA
#[derive(Debug, Clone)]
pub struct FastICA {
    pub n_components: usize,
    pub max_iters: usize,
    pub tol: f64,

    components: Vec<Vec<f64>>,
    mean: Vec<f64>,
    fitted: bool,
}

impl FastICA {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iters: 200,
            tol: 1e-4,
            components: Vec::new(),
            mean: Vec::new(),
            fitted: false,
        }
    }

    #[inline]
    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[inline]
    fn norm(v: &[f64]) -> f64 {
        Self::dot(v, v).sqrt()
    }

    fn normalize(v: &mut [f64]) {
        let n = Self::norm(v);
        for x in v.iter_mut() {
            *x /= n;
        }
    }

    #[inline]
    fn g(x: f64) -> f64 {
        x.tanh()
    }

    #[inline]
    fn g_deriv(x: f64) -> f64 {
        1.0 - x.tanh().powi(2)
    }
}

impl<D> FastICA
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit(&mut self, data: &D) {
        let n = data.len();
        let d = data.n_features();
        assert!(n > 0 && d > 0);
        assert!(self.n_components <= d);

        // center data
        self.mean = vec![0.0; d];
        for i in 0..n {
            let x = data.feature_row(i);
            for j in 0..d {
                self.mean[j] += x[j];
            }
        }
        for j in 0..d {
            self.mean[j] /= n as f64;
        }

        let mut x_centered: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut v = data.feature_row(i).to_vec();
            for j in 0..d {
                v[j] -= self.mean[j];
            }
            v
        })
        .collect();

        // initialize random components
        let mut rng = thread_rng();
        self.components.clear();

        for _ in 0..self.n_components {
            let mut w = vec![0.0; d];
            w.shuffle(&mut rng);
            Self::normalize(&mut w);
            self.components.push(w);
        }

        // FastICA fixed-point iteration
        for c in 0..self.n_components {
            let mut w = self.components[c].clone();

            for _ in 0..self.max_iters {
                let mut w_new = vec![0.0; d];
                let mut g_deriv_mean = 0.0;

                for i in 0..n {
                    let xi = &x_centered[i];
                    let wx = Self::dot(&w, xi);
                    let gx = Self::g(wx);
                    let gpx = Self::g_deriv(wx);

                    for j in 0..d {
                        w_new[j] += xi[j] * gx;
                    }
                    g_deriv_mean += gpx;
                }

                for j in 0..d {
                    w_new[j] = w_new[j] / n as f64
                    - g_deriv_mean / n as f64 * w[j];
                }

                // decorrelate
                for k in 0..c {
                    let proj = Self::dot(&w_new, &self.components[k]);
                    for j in 0..d {
                        w_new[j] -= proj * self.components[k][j];
                    }
                }

                Self::normalize(&mut w_new);

                let diff = (Self::dot(&w, &w_new).abs() - 1.0).abs();
                w = w_new;

                if diff < self.tol {
                    break;
                }
            }

            self.components[c] = w;
        }

        self.fitted = true;
    }

    pub fn transform(&self, data: &D) -> Vec<Vec<f64>> {
        assert!(self.fitted, "FastICA not fitted");

        (0..data.len())
        .map(|i| {
            let mut x = data.feature_row(i).to_vec();
            for j in 0..x.len() {
                x[j] -= self.mean[j];
            }
            self.components
            .iter()
            .map(|w| Self::dot(w, &x))
            .collect()
        })
        .collect()
    }
}
