// rusanta-ml/src/models/tree.rs

use crate::dataset::Dataset;
use crate::pipeline::Estimator;

/// Decision tree node.
#[derive(Debug, Clone)]
enum Node {
    Leaf { value: f64 },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

/// CART-style Decision Tree Regressor.
///
/// Design goals:
/// - numeric features
/// - numeric target
/// - greedy variance reduction
/// - correctness over performance
#[derive(Debug, Clone)]
pub struct DecisionTreeRegressor {
    pub max_depth: usize,
    pub min_samples_split: usize,
    root: Option<Node>,
}

impl DecisionTreeRegressor {
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            min_samples_split: 2,
            root: None,
        }
    }

    fn variance(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
    }

    fn best_split<D: Dataset<Feature = f64, Target = f64>>(
        &self,
        data: &D,
        indices: &[usize],
    ) -> Option<(usize, f64)> {
        let d = data.n_features();
        let mut best_var = f64::INFINITY;
        let mut best = None;

        for feature in 0..d {
            let mut values: Vec<f64> = indices
            .iter()
            .map(|&i| data.get_feature(i, feature))
            .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for w in values.windows(2) {
                let threshold = (w[0] + w[1]) / 2.0;

                let mut left = Vec::new();
                let mut right = Vec::new();

                for &i in indices {
                    if data.get_feature(i, feature) <= threshold {
                        left.push(data.get_target(i));
                    } else {
                        right.push(data.get_target(i));
                    }
                }

                if left.is_empty() || right.is_empty() {
                    continue;
                }

                let var = (left.len() as f64 * Self::variance(&left)
                + right.len() as f64 * Self::variance(&right))
                / indices.len() as f64;

                if var < best_var {
                    best_var = var;
                    best = Some((feature, threshold));
                }
            }
        }

        best
    }

    fn build<D: Dataset<Feature = f64, Target = f64>>(
        &self,
        data: &D,
        indices: &[usize],
        depth: usize,
    ) -> Node {
        let targets: Vec<f64> = indices.iter().map(|&i| data.get_target(i)).collect();
        let mean = targets.iter().sum::<f64>() / targets.len() as f64;

        if depth >= self.max_depth || indices.len() < self.min_samples_split {
            return Node::Leaf { value: mean };
        }

        if let Some((feature, threshold)) = self.best_split(data, indices) {
            let mut left_idx = Vec::new();
            let mut right_idx = Vec::new();

            for &i in indices {
                if data.get_feature(i, feature) <= threshold {
                    left_idx.push(i);
                } else {
                    right_idx.push(i);
                }
            }

            if left_idx.is_empty() || right_idx.is_empty() {
                return Node::Leaf { value: mean };
            }

            Node::Split {
                feature,
                threshold,
                left: Box::new(self.build(data, &left_idx, depth + 1)),
                right: Box::new(self.build(data, &right_idx, depth + 1)),
            }
        } else {
            Node::Leaf { value: mean }
        }
    }

    fn predict_one(&self, node: &Node, row: &[f64]) -> f64 {
        match node {
            Node::Leaf { value } => *value,
            Node::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if row[*feature] <= *threshold {
                    self.predict_one(left, row)
                } else {
                    self.predict_one(right, row)
                }
            }
        }
    }
}

impl<D> Estimator<D> for DecisionTreeRegressor
where
D: Dataset<Feature = f64, Target = f64>,
{
    fn fit(&mut self, data: &D) {
        let indices: Vec<usize> = (0..data.len()).collect();
        self.root = Some(self.build(data, &indices, 0));
    }

    fn predict(&self, data: &D) -> Vec<f64> {
        let root = self.root.as_ref().expect("model not fitted");
        (0..data.len())
        .map(|i| self.predict_one(root, data.feature_row(i)))
        .collect()
    }
}
