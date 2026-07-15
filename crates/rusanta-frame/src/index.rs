use crate::error::Result;
use rusanta_core::traits::numeric::Numeric;
use rusanta_core::traits::dataset::Dataset;

/// Base trait for evaluation metrics.
///
/// A metric compares predictions against ground truth
/// and returns a scalar score.
pub trait Metric<T>: Send + Sync
where
T: Numeric,
{
    /// Name of the metric.
    fn name(&self) -> &str;

    /// Compute metric value for a single prediction-target pair.
    fn compute<Y>(&self, preds: &[T], targets: &Y) -> Result<T>
    where
    Y: Dataset<T>;
}

/* ===========================
 * Regression metrics
 * =========================== */

/// Mean Squared Error (MSE).
pub struct MeanSquaredError;

impl<T> Metric<T> for MeanSquaredError
where
T: Numeric,
{
    fn name(&self) -> &str {
        "mean_squared_error"
    }

    fn compute<Y>(&self, preds: &[T], targets: &Y) -> Result<T>
    where
    Y: Dataset<T>,
    {
        let n = targets.len();
        if preds.len() != n {
            return Err(crate::error::Error::InvalidValue(
                "prediction/target length mismatch".into(),
            ));
        }

        let mut acc = T::zero();
        for i in 0..n {
            let diff = preds[i] - targets.get(i)?;
            acc = acc + diff * diff;
        }

        Ok(acc / T::from_usize(n))
    }
}

/// Mean Absolute Error (MAE).
pub struct MeanAbsoluteError;

impl<T> Metric<T> for MeanAbsoluteError
where
T: Numeric,
{
    fn name(&self) -> &str {
        "mean_absolute_error"
    }

    fn compute<Y>(&self, preds: &[T], targets: &Y) -> Result<T>
    where
    Y: Dataset<T>,
    {
        let n = targets.len();
        if preds.len() != n {
            return Err(crate::error::Error::InvalidValue(
                "prediction/target length mismatch".into(),
            ));
        }

        let mut acc = T::zero();
        for i in 0..n {
            let diff = preds[i] - targets.get(i)?;
            acc = acc + if diff < T::zero() { -diff } else { diff };
        }

        Ok(acc / T::from_usize(n))
    }
}

/* ===========================
 * Classification metrics
 * =========================== */

/// Accuracy metric.
///
/// Assumes preds contain class labels.
pub struct Accuracy;

impl<T> Metric<T> for Accuracy
where
T: Numeric + PartialEq,
{
    fn name(&self) -> &str {
        "accuracy"
    }

    fn compute<Y>(&self, preds: &[T], targets: &Y) -> Result<T>
    where
    Y: Dataset<T>,
    {
        let n = targets.len();
        if preds.len() != n {
            return Err(crate::error::Error::InvalidValue(
                "prediction/target length mismatch".into(),
            ));
        }

        let mut correct = 0usize;
        for i in 0..n {
            if preds[i] == targets.get(i)? {
                correct += 1;
            }
        }

        Ok(T::from_usize(correct) / T::from_usize(n))
    }
}
