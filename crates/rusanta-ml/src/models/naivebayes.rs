// rusanta-ml/src/models/naivebayes.rs
// Gaussian Naive Bayes — binary & multiclass, numeric features

use crate::dataset::Dataset;
use crate::pipeline::Estimator;
use std::collections::HashMap;

/// Gaussian Naive Bayes
#[derive(Debug, Clone)]
pub struct GaussianNaiveBayes {
    class_priors: HashMap<f64, f64>,
    mean: HashMap<f64, Vec<f64>>,
    var: HashMap<f64, Vec<f64>>,
    fitted: bool,
}

impl GaussianNaiveBayes {
    pub fn new() -> Self {
        Self {
            class_priors: HashMap::new(),
            mean: HashMap::new(),
            var: HashMap::new(),
            fitted: false,
        }
    }

    #[inline]
    fn gaussian_log_prob(x: f64, mean: f64, var: f64) -> f64 {
        let var = var.max(1e-9); // numerical stability
        -0.5 * ((x - mean).powi(2) / var + var.ln() + std::f64::consts::LN_2PI)
    }
}

impl<D> Estimator<D> for GaussianNaiveBayes
where
D: Dataset<Feature = f64, Target = f64>,
{
    fn fit(&mut self, data: &D) {
        let n = data.len();
        let d = data.n_features();

        assert!(n > 0 && d > 0);

        let mut class_indices: HashMap<f64, Vec<usize>> = HashMap::new();

        for i in 0..n {
            class_indices
            .entry(data.get_target(i))
            .or_default()
            .push(i);
        }

        self.class_priors.clear();
        self.mean.clear();
        self.var.clear();

        for (class, idxs) in class_indices.iter() {
            let count = idxs.len() as f64;
            self.class_priors.insert(*class, count / n as f64);

            let mut mean = vec![0.0; d];
            for &i in idxs {
                let x = data.feature_row(i);
                for j in 0..d {
                    mean[j] += x[j];
                }
            }
            for j in 0..d {
                mean[j] /= count;
            }

            let mut var = vec![0.0; d];
            for &i in idxs {
                let x = data.feature_row(i);
                for j in 0..d {
                    let diff = x[j] - mean[j];
                    var[j] += diff * diff;
                }
            }
            for j in 0..d {
                var[j] /= count;
            }

            self.mean.insert(*class, mean);
            self.var.insert(*class, var);
        }

        self.fitted = true;
    }

    fn predict(&self, data: &D) -> Vec<f64> {
        assert!(self.fitted, "GaussianNaiveBayes not fitted");

        (0..data.len())
        .map(|i| {
            let x = data.feature_row(i);

            self.class_priors
            .iter()
            .map(|(&class, &prior)| {
                let mean = &self.mean[&class];
                let var = &self.var[&class];

                let log_prob = prior.ln()
                + x.iter()
                .zip(mean.iter().zip(var.iter()))
                .map(|(&xi, (&m, &v))| {
                    Self::gaussian_log_prob(xi, m, v)
                })
                .sum::<f64>();

                (class, log_prob)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
        })
        .collect()
    }
}
