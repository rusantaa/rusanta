// rusanta-ml/src/models/bernoullinb.rs
// Bernoulli Naive Bayes — binary features, binary & multiclass targets

use crate::dataset::Dataset;
use crate::pipeline::Estimator;
use std::collections::HashMap;

/// Bernoulli Naive Bayes
///
/// Assumptions:
/// - features are binary (0.0 or 1.0)
/// - targets are categorical (f64 labels)
#[derive(Debug, Clone)]
pub struct BernoulliNaiveBayes {
    class_priors: HashMap<f64, f64>,
    feature_prob: HashMap<f64, Vec<f64>>, // P(x_j = 1 | class)
    fitted: bool,
}

impl BernoulliNaiveBayes {
    pub fn new() -> Self {
        Self {
            class_priors: HashMap::new(),
            feature_prob: HashMap::new(),
            fitted: false,
        }
    }
}

impl<D> Estimator<D> for BernoulliNaiveBayes
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
        self.feature_prob.clear();

        for (class, idxs) in class_indices.iter() {
            let count = idxs.len() as f64;
            self.class_priors.insert(*class, count / n as f64);

            // Laplace smoothing (alpha = 1)
            let mut probs = vec![1.0; d];
            let denom = count + 2.0;

            for &i in idxs {
                let x = data.feature_row(i);
                for j in 0..d {
                    if x[j] > 0.0 {
                        probs[j] += 1.0;
                    }
                }
            }

            for j in 0..d {
                probs[j] /= denom;
            }

            self.feature_prob.insert(*class, probs);
        }

        self.fitted = true;
    }

    fn predict(&self, data: &D) -> Vec<f64> {
        assert!(self.fitted, "BernoulliNaiveBayes not fitted");

        (0..data.len())
        .map(|i| {
            let x = data.feature_row(i);

            self.class_priors
            .iter()
            .map(|(&class, &prior)| {
                let probs = &self.feature_prob[&class];

                let log_prob = prior.ln()
                + x.iter()
                .zip(probs.iter())
                .map(|(&xi, &p)| {
                    if xi > 0.0 {
                        p.ln()
                    } else {
                        (1.0 - p).ln()
                    }
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
