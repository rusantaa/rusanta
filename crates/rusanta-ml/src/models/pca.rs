// rusanta-ml/src/models/pca.rs
// Principal Component Analysis (PCA)
// Eigen-decomposition via power iteration (no external BLAS)

use crate::dataset::Dataset;

/// Principal Component Analysis
#[derive(Debug, Clone)]
pub struct PCA {
    pub n_components: usize,
    pub max_iters: usize,
    pub tol: f64,

    pub components: Vec<Vec<f64>>,
    pub mean: Vec<f64>,
    fitted: bool,
}

impl PCA {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iters: 1000,
            tol: 1e-8,
            components: Vec::new(),
            mean: Vec::new(),
            fitted: false,
        }
    }

    #[inline]
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

    fn mat_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
        m.iter().map(|row| Self::dot(row, v)).collect()
    }
}

impl<D> PCA
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit(&mut self, data: &D) {
        let n = data.len();
        let d = data.n_features();
        assert!(n > 0 && d > 0);
        assert!(self.n_components <= d);

        // compute mean
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

        // center data
        let mut x: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut v = data.feature_row(i).to_vec();
            for j in 0..d {
                v[j] -= self.mean[j];
            }
            v
        })
        .collect();

        // covariance matrix
        let mut cov = vec![vec![0.0; d]; d];
        for i in 0..n {
            for j in 0..d {
                for k in 0..d {
                    cov[j][k] += x[i][j] * x[i][k];
                }
            }
        }
        for j in 0..d {
            for k in 0..d {
                cov[j][k] /= (n - 1) as f64;
            }
        }

        self.components.clear();

        // power iteration for top eigenvectors
        for _ in 0..self.n_components {
            let mut v = vec![1.0; d];
            Self::normalize(&mut v);

            for _ in 0..self.max_iters {
                let mut v_new = Self::mat_vec(&cov, &v);
                Self::normalize(&mut v_new);

                let diff = v
                .iter()
                .zip(v_new.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

                v = v_new;
                if diff < self.tol {
                    break;
                }
            }

            // deflation
            let lambda = Self::dot(&v, &Self::mat_vec(&cov, &v));
            for i in 0..d {
                for j in 0..d {
                    cov[i][j] -= lambda * v[i] * v[j];
                }
            }

            self.components.push(v);
        }

        self.fitted = true;
    }

    pub fn transform(&self, data: &D) -> Vec<Vec<f64>> {
        assert!(self.fitted);

        (0..data.len())
        .map(|i| {
            let mut x = data.feature_row(i).to_vec();
            for j in 0..x.len() {
                x[j] -= self.mean[j];
            }
            self.components
            .iter()
            .map(|c| Self::dot(c, &x))
            .collect()
        })
        .collect()
    }
}
