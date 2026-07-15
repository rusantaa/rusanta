// rusanta-ml/src/models/bagging.rs
// Bagging (Bootstrap Aggregating) — generic ensemble wrapper

use crate::dataset::Dataset;
use crate::pipeline::Estimator;
use rand::Rng;

/// Bagging ensemble
///
/// Works with any model implementing `Estimator + Clone`
#[derive(Debug, Clone)]
pub struct Bagging<M> {
    pub n_estimators: usize,
    pub sample_ratio: f64,
    estimators: Vec<M>,
    fitted: bool,
}

impl<M> Bagging<M>
where
M: Clone,
{
    pub fn new(base: M, n_estimators: usize) -> Self {
        Self {
            n_estimators,
            sample_ratio: 1.0,
            estimators: vec![base; n_estimators],
            fitted: false,
        }
    }
}

impl<M, D> Estimator<D> for Bagging<M>
where
M: Estimator<D> + Clone,
D: Dataset<Feature = f64, Target = f64>,
{
    fn fit(&mut self, data: &D) {
        let n = data.len();
        let m = ((n as f64) * self.sample_ratio).ceil() as usize;
        let mut rng = rand::thread_rng();

        for model in &mut self.estimators {
            // bootstrap indices
            let indices: Vec<usize> =
            (0..m).map(|_| rng.gen_range(0..n)).collect();

            let subset = data.subset(&indices);
            model.fit(&subset);
        }

        self.fitted = true;
    }

    fn predict(&self, data: &D) -> Vec<f64> {
        assert!(self.fitted, "Bagging not fitted");

        let n = data.len();
        let mut preds = vec![0.0; n];

        for model in &self.estimators {
            let p = model.predict(data);
            for i in 0..n {
                preds[i] += p[i];
            }
        }

        for i in 0..n {
            preds[i] /= self.estimators.len() as f64;
        }

        preds
    }
}
