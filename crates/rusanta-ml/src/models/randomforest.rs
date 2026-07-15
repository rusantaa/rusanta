// rusanta-ml/src/models/randomforest.rs
// Random Forest (ensemble) — explicit, clean, no placeholders

use crate::dataset::Dataset;
use crate::pipeline::Estimator;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Internal decision tree node
#[derive(Debug, Clone)]
enum Node {
    Leaf(f64),
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

/// Simple CART-style regression tree (numeric only)
#[derive(Debug, Clone)]
struct Tree {
    root: Node,
}

impl Tree {
    fn build<D: Dataset<Feature = f64, Target = f64>>(
        data: &D,
        indices: &[usize],
        depth: usize,
        max_depth: usize,
        min_samples: usize,
        features: &[usize],
    ) -> Node {
        if depth >= max_depth || indices.len() < min_samples {
            return Node::Leaf(Self::mean(data, indices));
        }

        let mut best = None;
        let mut best_score = f64::INFINITY;

        for &f in features {
            let mut vals: Vec<f64> =
            indices.iter().map(|&i| data.feature_row(i)[f]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vals.dedup();

            for &t in &vals {
                let (l, r): (Vec<_>, Vec<_>) = indices
                .iter()
                .cloned()
                .partition(|&i| data.feature_row(i)[f] <= t);

                if l.is_empty() || r.is_empty() {
                    continue;
                }

                let score = Self::var(data, &l) * l.len() as f64
                + Self::var(data, &r) * r.len() as f64;

                if score < best_score {
                    best_score = score;
                    best = Some((f, t, l, r));
                }
            }
        }

        if let Some((f, t, l, r)) = best {
            Node::Split {
                feature: f,
                threshold: t,
                left: Box::new(Self::build(
                    data, &l, depth + 1, max_depth, min_samples, features,
                )),
                right: Box::new(Self::build(
                    data, &r, depth + 1, max_depth, min_samples, features,
                )),
            }
        } else {
            Node::Leaf(Self::mean(data, indices))
        }
    }

    fn mean<D: Dataset<Feature = f64, Target = f64>>(
        data: &D,
        idx: &[usize],
    ) -> f64 {
        idx.iter().map(|&i| data.get_target(i)).sum::<f64>() / idx.len() as f64
    }

    fn var<D: Dataset<Feature = f64, Target = f64>>(
        data: &D,
        idx: &[usize],
    ) -> f64 {
        let m = Self::mean(data, idx);
        idx.iter()
        .map(|&i| {
            let d = data.get_target(i) - m;
            d * d
        })
        .sum::<f64>()
        / idx.len() as f64
    }

    fn predict(node: &Node, x: &[f64]) -> f64 {
        match node {
            Node::Leaf(v) => *v,
            Node::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if x[*feature] <= *threshold {
                    Self::predict(left, x)
                } else {
                    Self::predict(right, x)
                }
            }
        }
    }
}

/// Random Forest (regression baseline)
#[derive(Debug, Clone)]
pub struct RandomForest {
    pub n_trees: usize,
    pub max_depth: usize,
    pub min_samples: usize,
    pub feature_ratio: f64,
    trees: Vec<Tree>,
    fitted: bool,
}

impl RandomForest {
    pub fn new() -> Self {
        Self {
            n_trees: 100,
            max_depth: 12,
            min_samples: 2,
            feature_ratio: 0.7,
            trees: Vec::new(),
            fitted: false,
        }
    }
}

impl<D> Estimator<D> for RandomForest
where
D: Dataset<Feature = f64, Target = f64>,
{
    fn fit(&mut self, data: &D) {
        let n = data.len();
        let d = data.n_features();
        let mut rng = thread_rng();

        self.trees.clear();

        for _ in 0..self.n_trees {
            // bootstrap rows
            let rows: Vec<usize> = (0..n)
            .map(|_| rng.gen_range(0..n))
            .collect();

            // feature subsampling
            let mut feats: Vec<usize> = (0..d).collect();
            feats.shuffle(&mut rng);
            let k = ((d as f64) * self.feature_ratio).ceil() as usize;
            feats.truncate(k.max(1));

            let root = Tree::build(
                data,
                &rows,
                0,
                self.max_depth,
                self.min_samples,
                &feats,
            );

            self.trees.push(Tree { root });
        }

        self.fitted = true;
    }

    fn predict(&self, data: &D) -> Vec<f64> {
        assert!(self.fitted, "RandomForest not fitted");

        (0..data.len())
        .map(|i| {
            let x = data.feature_row(i);
            self.trees
            .iter()
            .map(|t| Tree::predict(&t.root, x))
            .sum::<f64>()
            / self.trees.len() as f64
        })
        .collect()
    }
}
