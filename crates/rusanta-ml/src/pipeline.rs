// rusanta-ml/src/pipeline.rs

use crate::dataset::Dataset;

/// A preprocessing step in a pipeline.
///
/// This mirrors sklearn’s Transformer API.
pub trait Transformer<D: Dataset> {
    /// Fit the transformer on a dataset.
    fn fit(&mut self, data: &D);

    /// Transform the dataset and return a new one.
    fn transform(&self, data: &D) -> D;

    /// Fit + transform shortcut.
    fn fit_transform(&mut self, data: &D) -> D {
        self.fit(data);
        self.transform(data)
    }
}

/// A trainable estimator (model).
///
/// This is intentionally minimal and decoupled from preprocessing.
pub trait Estimator<D: Dataset> {
    /// Fit the estimator.
    fn fit(&mut self, data: &D);

    /// Predict targets for a dataset.
    fn predict(&self, data: &D) -> Vec<D::Target>;
}

/// A machine learning pipeline.
///
/// A pipeline is:
/// transformers → transformers → model
///
/// This guarantees:
/// - consistent preprocessing
/// - reproducible training/inference
pub struct Pipeline<D: Dataset, M: Estimator<D>> {
    transformers: Vec<Box<dyn Transformer<D>>>,
    model: M,
}

impl<D, M> Pipeline<D, M>
where
D: Dataset,
M: Estimator<D>,
{
    /// Create a new pipeline.
    pub fn new(model: M) -> Self {
        Self {
            transformers: Vec::new(),
            model,
        }
    }

    /// Add a transformer step.
    pub fn add_transformer<T>(&mut self, t: T)
    where
    T: Transformer<D> + 'static,
    {
        self.transformers.push(Box::new(t));
    }

    /// Fit the entire pipeline.
    pub fn fit(&mut self, mut data: D) {
        for t in self.transformers.iter_mut() {
            data = t.fit_transform(&data);
        }
        self.model.fit(&data);
    }

    /// Predict using the fitted pipeline.
    pub fn predict(&self, mut data: D) -> Vec<D::Target> {
        for t in self.transformers.iter() {
            data = t.transform(&data);
        }
        self.model.predict(&data)
    }
}
