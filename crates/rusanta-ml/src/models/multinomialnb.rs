// rusanta-ml/src/models/multinomialnb.rs
// Multinomial Naive Bayes
// Suitable for count-based features (text, bag-of-words)

use crate::dataset::Dataset;
use std::collections::HashMap;

/// Multinomial Naive Bayes
#[derive(Debug, Clone)]
pub struct MultinomialNB {
    pub alpha: f64, // Laplace smoothing

    class_log_prior: Vec<f64>,
    feature_log_prob: Vec<Vec<f64>>,
    classes: Vec<f64>,
    fitted: bool,
}

impl MultinomialNB {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            class_log_prior: Vec::new(),
            feature_log_prob: Vec::new(),
            classes: Vec::new(),
            fitted: false,
        }
    }
}

impl<D> MultinomialNB
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit(&mut self, data: &D) {
        let n = data.len();
        let d = data.n_features();

        // map class labels to indices
        let mut class_map: HashMap<f64, usize> = HashMap::new();
        for i in 0..n {
            let y = data.target(i);
            class_map.entry(y).or_insert_with(|| class_map.len());
        }

        let n_classes = class_map.len();
        self.classes = vec![0.0; n_classes];
        for (k, v) in class_map.iter() {
            self.classes[*v] = *k;
        }

        let mut class_count = vec![0.0; n_classes];
        let mut feature_count = vec![vec![0.0; d]; n_classes];

        for i in 0..n {
            let x = data.feature_row(i);
            let y = data.target(i);
            let c = class_map[&y];

            class_count[c] += 1.0;
            for j in 0..d {
                feature_count[c][j] += x[j];
            }
        }

        // log class priors
        self.class_log_prior = class_count
        .iter()
        .map(|&c| (c / n as f64).ln())
        .collect();

        // log feature probabilities
        self.feature_log_prob = Vec::with_capacity(n_classes);
        for c in 0..n_classes {
            let sum_fc: f64 = feature_count[c].iter().sum::<f64>()
            + self.alpha * d as f64;

            let probs = feature_count[c]
            .iter()
            .map(|&f| ((f + self.alpha) / sum_fc).ln())
            .collect();

            self.feature_log_prob.push(probs);
        }

        self.fitted = true;
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        assert!(self.fitted);

        let mut best_class = 0;
        let mut best_score = f64::NEG_INFINITY;

        for (c, log_prior) in self.class_log_prior.iter().enumerate() {
            let mut score = *log_prior;
            for j in 0..x.len() {
                score += x[j] * self.feature_log_prob[c][j];
            }
            if score > best_score {
                best_score = score;
                best_class = c;
            }
        }

        self.classes[best_class]
    }

    pub fn predict_batch(&self, data: &D) -> Vec<f64> {
        (0..data.len())
        .map(|i| self.predict(data.feature_row(i)))
        .collect()
    }
}
