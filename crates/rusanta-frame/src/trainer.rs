use crate::dataloader::DataLoader;
use crate::error::Result;
use crate::metrics::Metric;
use rusanta_core::traits::numeric::Numeric;
use rusanta_core::traits::dataset::Dataset;
use rusanta_core::traits::model::{Estimator, Predictor, IsFitted, IsUnfitted};

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub epochs: usize,
    pub verbose: bool,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            verbose: true,
        }
    }
}

/// Trainer orchestrates the training loop.
///
/// It does NOT:
/// - define loss functions
/// - define optimizers
/// - assume specific data layout
///
/// It ONLY:
/// - iterates epochs
/// - iterates batches
/// - calls model APIs
pub struct Trainer;

impl Trainer {
    /// Train a model using a DataLoader.
    ///
    /// This mirrors sklearn-style `.fit()`,
    /// but with explicit epochs and batching.
    pub fn fit<T, X, Y, M>(
        mut model: M,
        mut loader: DataLoader<(X, Y)>,
                           config: TrainerConfig,
    ) -> Result<M>
    where
    T: Numeric,
    X: Dataset<T>,
    Y: Dataset<T>,
    M: Estimator<T, X, Y, IsUnfitted>,
    {
        for epoch in 0..config.epochs {
            if config.verbose {
                println!("Epoch {}/{}", epoch + 1, config.epochs);
            }

            loader.reset();

            for batch in loader.by_ref() {
                let batch = batch?;
                for (x, y) in batch {
                    model = model.fit(&x, &y)?;
                }
            }
        }

        Ok(model)
    }

    /// Evaluate a fitted model using a DataLoader and metrics.
    pub fn evaluate<T, X, Y, M>(
        model: &M,
        mut loader: DataLoader<(X, Y)>,
                                metrics: &[Box<dyn Metric<T>>],
    ) -> Result<Vec<T>>
    where
    T: Numeric,
    X: Dataset<T>,
    Y: Dataset<T>,
    M: Predictor<T, X, IsFitted>,
    {
        let mut results = vec![T::zero(); metrics.len()];
        let mut counts = 0usize;

        loader.reset();

        for batch in loader.by_ref() {
            let batch = batch?;
            for (x, y) in batch {
                let preds = model.predict(&x)?;
                for (i, metric) in metrics.iter().enumerate() {
                    results[i] = results[i] + metric.compute(&preds, &y)?;
                }
                counts += 1;
            }
        }

        if counts > 0 {
            for r in &mut results {
                *r = *r / T::from_usize(counts);
            }
        }

        Ok(results)
    }
}
