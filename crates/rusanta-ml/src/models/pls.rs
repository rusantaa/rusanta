// rusanta-ml/src/models/pls.rs
// Partial Least Squares Regression (PLS1 – single target)
// NIPALS algorithm

use crate::dataset::Dataset;

/// Partial Least Squares Regression (PLS1)
#[derive(Debug, Clone)]
pub struct PLS {
    pub n_components: usize,
    pub max_iters: usize,
    pub tol: f64,

    pub x_weights: Vec<Vec<f64>>,
    pub y_weights: Vec<f64>,
    pub coef: Vec<f64>,
    pub intercept: f64,

    fitted: bool,
}

impl PLS {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iters: 500,
            tol: 1e-6,
            x_weights: Vec::new(),
            y_weights: Vec::new(),
            coef: Vec::new(),
            intercept: 0.0,
            fitted: false,
        }
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn norm(v: &[f64]) -> f64 {
        Self::dot(v, v).sqrt()
    }

    fn normalize(v: &mut [f64]) {
        let n = Self::norm(v);
        if n > 0.0 {
            for x in v.iter_mut() {
                *x /= n;
            }
        }
    }

    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

impl<D> PLS
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit(&mut self, data: &D) {
        let n = data.len();
        let p = data.n_features();
        assert!(n > 0 && p > 0);

        let mut x: Vec<Vec<f64>> =
        (0..n).map(|i| data.feature_row(i).to_vec()).collect();
        let mut y: Vec<f64> = (0..n).map(|i| data.target(i)).collect();

        // center X and y
        let x_mean: Vec<f64> = (0..p)
        .map(|j| Self::mean(&x.iter().map(|r| r[j]).collect::<Vec<_>>()))
        .collect();
        let y_mean = Self::mean(&y);

        for i in 0..n {
            for j in 0..p {
                x[i][j] -= x_mean[j];
            }
            y[i] -= y_mean;
        }

        self.x_weights.clear();
        self.y_weights.clear();

        for _ in 0..self.n_components {
            // initialize u as y
            let mut u = y.clone();
            let mut w = vec![0.0; p];
            let mut t = vec![0.0; n];
            let mut q = 0.0;

            for _ in 0..self.max_iters {
                // w = Xᵀ u
                for j in 0..p {
                    w[j] = (0..n).map(|i| x[i][j] * u[i]).sum();
                }
                Self::normalize(&mut w);

                // t = X w
                for i in 0..n {
                    t[i] = Self::dot(&x[i], &w);
                }

                // q = yᵀ t / (tᵀ t)
                let t_norm_sq = Self::dot(&t, &t);
                q = (0..n).map(|i| y[i] * t[i]).sum::<f64>() / t_norm_sq;

                // u_new = y q
                let mut u_new: Vec<f64> = y.iter().map(|yi| yi * q).collect();

                let diff = Self::dot(
                    &u.iter().zip(u_new.iter()).map(|(a, b)| a - b).collect::<Vec<_>>(),
                                     &u.iter().zip(u_new.iter()).map(|(a, b)| a - b).collect::<Vec<_>>(),
                )
                .sqrt();

                u = u_new;
                if diff < self.tol {
                    break;
                }
            }

            // deflation
            let t_norm_sq = Self::dot(&t, &t);
            for i in 0..n {
                for j in 0..p {
                    x[i][j] -= t[i] * w[j];
                }
                y[i] -= t[i] * q;
            }

            self.x_weights.push(w);
            self.y_weights.push(q);
        }

        // compute regression coefficients
        self.coef = vec![0.0; p];
        for (k, w) in self.x_weights.iter().enumerate() {
            for j in 0..p {
                self.coef[j] += w[j] * self.y_weights[k];
            }
        }

        self.intercept = y_mean
        - x_mean
        .iter()
        .zip(self.coef.iter())
        .map(|(m, b)| m * b)
        .sum::<f64>();

        self.fitted = true;
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        assert!(self.fitted);
        self.intercept + Self::dot(&self.coef, x)
    }
}
