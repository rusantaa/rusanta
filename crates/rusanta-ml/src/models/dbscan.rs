// rusanta-ml/src/models/dbscan.rs
// DBSCAN — density-based clustering, full implementation

use crate::dataset::Dataset;
use std::collections::VecDeque;

/// DBSCAN clustering
#[derive(Debug, Clone)]
pub struct DBSCAN {
    pub eps: f64,
    pub min_samples: usize,
    labels: Vec<i32>, // -1 = noise, >=0 = cluster id
    fitted: bool,
}

impl DBSCAN {
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Self {
            eps,
            min_samples,
            labels: Vec::new(),
            fitted: false,
        }
    }

    fn distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
    }

    fn region_query<D: Dataset<Feature = f64, Target = f64>>(
        data: &D,
        point: usize,
        eps: f64,
    ) -> Vec<usize> {
        let x = data.feature_row(point);
        (0..data.len())
        .filter(|&i| Self::distance(x, data.feature_row(i)) <= eps)
        .collect()
    }
}

impl<D> DBSCAN
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit(&mut self, data: &D) {
        let n = data.len();
        assert!(n > 0);

        self.labels = vec![-2; n]; // -2 = unvisited
        let mut cluster_id = 0;

        for i in 0..n {
            if self.labels[i] != -2 {
                continue;
            }

            let neighbors = Self::region_query(data, i, self.eps);

            if neighbors.len() < self.min_samples {
                self.labels[i] = -1; // noise
                continue;
            }

            // start new cluster
            self.labels[i] = cluster_id;
            let mut queue: VecDeque<usize> = neighbors.into();

            while let Some(p) = queue.pop_front() {
                if self.labels[p] == -1 {
                    self.labels[p] = cluster_id;
                }
                if self.labels[p] != -2 {
                    continue;
                }

                self.labels[p] = cluster_id;
                let nbrs = Self::region_query(data, p, self.eps);

                if nbrs.len() >= self.min_samples {
                    for q in nbrs {
                        if self.labels[q] == -2 {
                            queue.push_back(q);
                        }
                    }
                }
            }

            cluster_id += 1;
        }

        self.fitted = true;
    }

    pub fn labels(&self) -> &[i32] {
        assert!(self.fitted, "DBSCAN not fitted");
        &self.labels
    }
}
