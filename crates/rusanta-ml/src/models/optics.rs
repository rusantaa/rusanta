// rusanta-ml/src/models/optics.rs
// OPTICS — Ordering Points To Identify the Clustering Structure
// Core-distance + reachability-distance implementation (numeric, euclidean)

use crate::dataset::Dataset;
use std::cmp::Ordering;

/// OPTICS clustering
#[derive(Debug, Clone)]
pub struct OPTICS {
    pub eps: f64,
    pub min_samples: usize,

    ordering: Vec<usize>,
    reachability: Vec<f64>,
    fitted: bool,
}

impl OPTICS {
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Self {
            eps,
            min_samples,
            ordering: Vec::new(),
            reachability: Vec::new(),
            fitted: false,
        }
    }

    #[inline]
    fn dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
    }

    fn neighbors<D: Dataset<Feature = f64, Target = f64>>(
        data: &D,
        i: usize,
        eps: f64,
    ) -> Vec<(usize, f64)> {
        let xi = data.feature_row(i);
        (0..data.len())
        .filter_map(|j| {
            let d = Self::dist(xi, data.feature_row(j));
            if d <= eps {
                Some((j, d))
            } else {
                None
            }
        })
        .collect()
    }

    fn core_distance(neighbors: &mut Vec<(usize, f64)>, min_samples: usize) -> Option<f64> {
        if neighbors.len() < min_samples {
            return None;
        }
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        Some(neighbors[min_samples - 1].1)
    }
}

impl<D> OPTICS
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit(&mut self, data: &D) {
        let n = data.len();
        assert!(n > 0);

        let mut processed = vec![false; n];
        self.reachability = vec![f64::INFINITY; n];
        self.ordering.clear();

        for i in 0..n {
            if processed[i] {
                continue;
            }

            let mut nbrs = Self::neighbors(data, i, self.eps);
            processed[i] = true;
            self.ordering.push(i);

            if let Some(core_dist) = Self::core_distance(&mut nbrs, self.min_samples) {
                let mut seeds: Vec<(usize, f64)> = Vec::new();

                for (j, d) in nbrs {
                    if !processed[j] {
                        let new_rd = core_dist.max(d);
                        self.reachability[j] = self.reachability[j].min(new_rd);
                        seeds.push((j, self.reachability[j]));
                    }
                }

                while !seeds.is_empty() {
                    seeds.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    let (q, _) = seeds.remove(0);

                    if processed[q] {
                        continue;
                    }

                    let mut qnbrs = Self::neighbors(data, q, self.eps);
                    processed[q] = true;
                    self.ordering.push(q);

                    if let Some(q_core) =
                        Self::core_distance(&mut qnbrs, self.min_samples)
                        {
                            for (r, d) in qnbrs {
                                if !processed[r] {
                                    let new_rd = q_core.max(d);
                                    if new_rd < self.reachability[r] {
                                        self.reachability[r] = new_rd;
                                        seeds.push((r, new_rd));
                                    }
                                }
                            }
                        }
                }
            }
        }

        self.fitted = true;
    }

    pub fn ordering(&self) -> &[usize] {
        assert!(self.fitted, "OPTICS not fitted");
        &self.ordering
    }

    pub fn reachability(&self) -> &[f64] {
        assert!(self.fitted, "OPTICS not fitted");
        &self.reachability
    }
}
