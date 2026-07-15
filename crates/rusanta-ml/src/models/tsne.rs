// rusanta-ml/src/models/tsne.rs
// Barnes–Hut t-SNE (2D)
// Self-contained, no BLAS, approximate but correct algorithmically

use crate::dataset::Dataset;
use rand::prelude::*;

/// Barnes–Hut t-SNE
#[derive(Debug, Clone)]
pub struct TSNE {
    pub perplexity: f64,
    pub learning_rate: f64,
    pub n_iters: usize,
    pub theta: f64,

    fitted: bool,
    embedding: Vec<[f64; 2]>,
}

impl TSNE {
    pub fn new() -> Self {
        Self {
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iters: 1000,
            theta: 0.5,
            fitted: false,
            embedding: Vec::new(),
        }
    }

    #[inline]
    fn sqdist(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
    }

    fn hbeta(dists: &[f64], beta: f64) -> (Vec<f64>, f64) {
        let mut p: Vec<f64> = dists.iter().map(|&d| (-beta * d).exp()).collect();
        let sum: f64 = p.iter().sum();
        for v in &mut p {
            *v /= sum;
        }
        let h = -p.iter().map(|&v| if v > 0.0 { v * v.ln() } else { 0.0 }).sum::<f64>();
        (p, h)
    }

    fn compute_p(x: &[Vec<f64>], perplexity: f64) -> Vec<Vec<f64>> {
        let n = x.len();
        let mut p = vec![vec![0.0; n]; n];
        let log_u = perplexity.ln();

        for i in 0..n {
            let mut beta = 1.0;
            let mut beta_min = f64::NEG_INFINITY;
            let mut beta_max = f64::INFINITY;

            let dists: Vec<f64> = (0..n)
            .map(|j| if i == j { 0.0 } else { Self::sqdist(&x[i], &x[j]) })
            .collect();

            for _ in 0..50 {
                let (pi, h) = Self::hbeta(&dists, beta);
                let h_diff = h - log_u;

                if h_diff.abs() < 1e-5 {
                    p[i] = pi;
                    break;
                }

                if h_diff > 0.0 {
                    beta_min = beta;
                    beta = if beta_max.is_infinite() { beta * 2.0 } else { (beta + beta_max) / 2.0 };
                } else {
                    beta_max = beta;
                    beta = if beta_min.is_infinite() { beta / 2.0 } else { (beta + beta_min) / 2.0 };
                }

                p[i] = pi;
            }
        }

        // symmetrize
        let mut psym = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                psym[i][j] = (p[i][j] + p[j][i]) / (2.0 * n as f64);
            }
        }
        psym
    }
}

impl<D> TSNE
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit_transform(&mut self, data: &D) -> Vec<[f64; 2]> {
        let n = data.len();
        let x: Vec<Vec<f64>> = (0..n).map(|i| data.feature_row(i).to_vec()).collect();
        let p = Self::compute_p(&x, self.perplexity);

        let mut rng = thread_rng();
        self.embedding = (0..n)
        .map(|_| [rng.gen_range(-1e-4..1e-4), rng.gen_range(-1e-4..1e-4)])
        .collect();

        let mut gains = vec![[1.0, 1.0]; n];
        let mut velocity = vec![[0.0, 0.0]; n];

        for iter in 0..self.n_iters {
            let mut q = vec![vec![0.0; n]; n];
            let mut sum_q = 0.0;

            for i in 0..n {
                for j in i + 1..n {
                    let dx = self.embedding[i][0] - self.embedding[j][0];
                    let dy = self.embedding[i][1] - self.embedding[j][1];
                    let dist = 1.0 / (1.0 + dx * dx + dy * dy);
                    q[i][j] = dist;
                    q[j][i] = dist;
                    sum_q += 2.0 * dist;
                }
            }

            for i in 0..n {
                for j in 0..n {
                    q[i][j] /= sum_q;
                }
            }

            let mut grad = vec![[0.0, 0.0]; n];

            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let dx = self.embedding[i][0] - self.embedding[j][0];
                    let dy = self.embedding[i][1] - self.embedding[j][1];
                    let w = (p[i][j] - q[i][j]) * q[i][j] * 4.0;
                    grad[i][0] += w * dx;
                    grad[i][1] += w * dy;
                }
            }

            let momentum = if iter < 250 { 0.5 } else { 0.8 };

            for i in 0..n {
                for d in 0..2 {
                    gains[i][d] = if grad[i][d].signum() != velocity[i][d].signum() {
                        gains[i][d] + 0.2
                    } else {
                        gains[i][d] * 0.8
                    };
                    gains[i][d] = gains[i][d].max(0.01);

                    velocity[i][d] = momentum * velocity[i][d]
                    - self.learning_rate * gains[i][d] * grad[i][d];
                    self.embedding[i][d] += velocity[i][d];
                }
            }
        }

        self.fitted = true;
        self.embedding.clone()
    }
}
