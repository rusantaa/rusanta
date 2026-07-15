// rusanta-ml/src/models/hierarchical.rs
// Hierarchical Agglomerative Clustering (HAC)
// Linkage: single, complete, average (UPGMA)
// Numeric features, Euclidean distance

use crate::dataset::Dataset;

/// Linkage strategy
#[derive(Debug, Clone, Copy)]
pub enum Linkage {
    Single,
    Complete,
    Average,
}

/// Hierarchical Agglomerative Clustering
#[derive(Debug, Clone)]
pub struct HierarchicalClustering {
    pub linkage: Linkage,
    pub n_clusters: usize,

    labels: Vec<usize>,
    fitted: bool,
}

impl HierarchicalClustering {
    pub fn new(n_clusters: usize, linkage: Linkage) -> Self {
        Self {
            linkage,
            n_clusters,
            labels: Vec::new(),
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

    fn cluster_distance(
        &self,
        ca: &[usize],
        cb: &[usize],
        data: &impl Dataset<Feature = f64, Target = f64>,
    ) -> f64 {
        match self.linkage {
            Linkage::Single => ca
            .iter()
            .flat_map(|&i| cb.iter().map(move |&j| Self::dist(
                data.feature_row(i),
                                                              data.feature_row(j),
            )))
            .fold(f64::INFINITY, f64::min),

            Linkage::Complete => ca
            .iter()
            .flat_map(|&i| cb.iter().map(move |&j| Self::dist(
                data.feature_row(i),
                                                              data.feature_row(j),
            )))
            .fold(0.0, f64::max),

            Linkage::Average => {
                let mut sum = 0.0;
                let mut cnt = 0;
                for &i in ca {
                    for &j in cb {
                        sum += Self::dist(
                            data.feature_row(i),
                                          data.feature_row(j),
                        );
                        cnt += 1;
                    }
                }
                sum / cnt as f64
            }
        }
    }
}

impl<D> HierarchicalClustering
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn fit(&mut self, data: &D) {
        let n = data.len();
        assert!(n > 0);
        assert!(self.n_clusters > 0 && self.n_clusters <= n);

        // start with each point as its own cluster
        let mut clusters: Vec<Vec<usize>> =
        (0..n).map(|i| vec![i]).collect();

        while clusters.len() > self.n_clusters {
            let mut best_i = 0;
            let mut best_j = 1;
            let mut best_dist = f64::INFINITY;

            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    let d = self.cluster_distance(
                        &clusters[i],
                        &clusters[j],
                        data,
                    );
                    if d < best_dist {
                        best_dist = d;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            // merge j into i
            let mut merged = clusters[best_i].clone();
            merged.extend_from_slice(&clusters[best_j]);

            clusters[best_i] = merged;
            clusters.remove(best_j);
        }

        // assign labels
        self.labels = vec![0; n];
        for (cid, cluster) in clusters.iter().enumerate() {
            for &i in cluster {
                self.labels[i] = cid;
            }
        }

        self.fitted = true;
    }

    pub fn labels(&self) -> &[usize] {
        assert!(self.fitted, "HierarchicalClustering not fitted");
        &self.labels
    }
}
