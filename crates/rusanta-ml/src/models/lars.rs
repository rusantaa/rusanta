// rusanta-ml/src/models/lars.rs
// Least Angle Regression (LARS)
// Batch algorithm, numeric stability prioritized over micro-optimizations

use crate::dataset::Dataset;

/// Least Angle Regression
#[derive(Debug, Clone)]
pub struct LARS {
    pub max_steps: usize,
    pub tol: f64,

    pub coef: Vec<f64>,
    pub intercept: f64,
    fitted: bool,
}

impl LARS {
    pub fn new() -> Self {
        Self {
            max_steps: 500,
            tol: 1e-8,
            coef: Vec::new(),
            intercept: 0.0,
            fitted: false,
        }
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }

    fn l2_norm(v: &[f64]) -> f64 {
        Self::dot(v, v).sqrt()
    }
}

impl<D> LARS
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit(&mut self, data: &D) {
        let n = data.len();
        let p = data.n_features();
        assert!(n > 0 && p > 0);

        let mut x: Vec<Vec<f64>> = (0..n)
        .map(|i| data.feature_row(i).to_vec())
        .collect();

        let mut y: Vec<f64> = (0..n).map(|i| data.target(i)).collect();

        // center X and y
        let x_means: Vec<f64> = (0..p)
        .map(|j| Self::mean(&x.iter().map(|r| r[j]).collect::<Vec<_>>()))
        .collect();
        let y_mean = Self::mean(&y);

        for i in 0..n {
            for j in 0..p {
                x[i][j] -= x_means[j];
            }
            y[i] -= y_mean;
        }

        let mut beta = vec![0.0; p];
        let mut active: Vec<usize> = Vec::new();
        let mut inactive: Vec<usize> = (0..p).collect();

        let mut mu = vec![0.0; n];

        for _ in 0..self.max_steps {
            // correlations
            let mut c = vec![0.0; p];
            for j in 0..p {
                c[j] = (0..n)
                .map(|i| x[i][j] * (y[i] - mu[i]))
                .sum::<f64>()
                .abs();
            }

            let (j_max, c_max) = c
            .iter()
            .enumerate()
            .filter(|(j, _)| inactive.contains(j))
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

            if c_max < self.tol {
                break;
            }

            active.push(j_max);
            inactive.retain(|&j| j != j_max);

            // compute equiangular direction
            let k = active.len();
            let mut g = vec![vec![0.0; k]; k];
            for i in 0..k {
                for j in 0..k {
                    let ai = active[i];
                    let aj = active[j];
                    g[i][j] = (0..n).map(|r| x[r][ai] * x[r][aj]).sum();
                }
            }

            // naive inversion (small k)
            let mut inv = vec![vec![0.0; k]; k];
            for i in 0..k {
                inv[i][i] = 1.0 / g[i][i];
            }

            let mut a = vec![0.0; n];
            for i in 0..n {
                for (idx, &j) in active.iter().enumerate() {
                    a[i] += x[i][j] * inv[idx][idx];
                }
            }

            let a_norm = Self::l2_norm(&a);
            for i in 0..n {
                a[i] /= a_norm;
            }

            let gamma = c_max / a_norm;

            for i in 0..n {
                mu[i] += gamma * a[i];
            }

            for (idx, &j) in active.iter().enumerate() {
                beta[j] += gamma * inv[idx][idx];
            }
        }

        self.coef = beta;
        self.intercept = y_mean
        - x_means
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
