// rusanta-ml/src/models/kmeans.rs
// K-Means clustering — Lloyd's algorithm, full implementation

use crate::dataset::Dataset;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// K-Means clustering
#[derive(Debug, Clone)]
pub struct KMeans {
    pub k: usize,
    pub max_iters: usize,
    pub tol: f64,
    centroids: Vec<Vec<f64>>,
    fitted: bool,
}

impl KMeans {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iters: 300,
            tol: 1e-4,
            centroids: Vec::new(),
            fitted: false,
        }
    }

    fn euclidean(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
    }

    fn mean(points: &[Vec<f64>]) -> Vec<f64> {
        let d = points[0].len();
        let mut m = vec![0.0; d];

        for p in points {
            for j in 0..d {
                m[j] += p[j];
            }
        }
        for j in 0..d {
            m[j] /= points.len() as f64;
        }
        m
    }
}

impl<D> KMeans
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit(&mut self, data: &D) {
        let n = data.len();
        let d = data.n_features();
        assert!(self.k > 0 && self.k <= n);

        let mut rng = thread_rng();

        // init centroids from random samples
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        self.centroids = indices[..self.k]
        .iter()
        .map(|&i| data.feature_row(i).to_vec())
        .collect();

        for _ in 0..self.max_iters {
            let mut clusters: Vec<Vec<Vec<f64>>> = vec![Vec::new(); self.k];

            for i in 0..n {
                let x = data.feature_row(i);
                let (cid, _) = self
                .centroids
                .iter()
                .enumerate()
                .map(|(j, c)| (j, Self::euclidean(x, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

                clusters[cid].push(x.to_vec());
            }

            let mut shift = 0.0;
            for j in 0..self.k {
                if clusters[j].is_empty() {
                    continue;
                }
                let new_c = Self::mean(&clusters[j]);
                shift += Self::euclidean(&self.centroids[j], &new_c);
                self.centroids[j] = new_c;
            }

            if shift < self.tol {
                break;
            }
        }

        self.fitted = true;
    }

    pub fn predict(&self, data: &D) -> Vec<usize> {
        assert!(self.fitted, "KMeans not fitted");

        (0..data.len())
        .map(|i| {
            let x = data.feature_row(i);
            self.centroids
            .iter()
            .enumerate()
            .map(|(j, c)| (j, Self::euclidean(x, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
        })
        .collect()
    }
}
