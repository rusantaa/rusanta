use crate::error::Result;
use crate::traits::dataset::Dataset;
use crate::traits::numeric::Numeric;

/// Marker trait for fitted state.
pub trait Fitted {}

/// Marker trait for unfitted state.
pub trait Unfitted {}

/// Zero-sized types representing model state.
pub struct IsFitted;
pub struct IsUnfitted;

impl Fitted for IsFitted {}
impl Unfitted for IsUnfitted {}

/// Base trait for all models.
///
/// This trait is intentionally empty.
/// It exists for documentation, bounds, and future extension.
pub trait Model {}

/// Trait for trainable models.
///
/// This is the equivalent of `fit()` in scikit-learn.
pub trait Estimator<T, X, Y, State>: Model
where
T: Numeric,
X: Dataset<T>,
Y: Dataset<T>,
{
    /// Fits the model to input data.
    ///
    /// Returns a new model in the fitted state.
    fn fit(self, x: &X, y: &Y) -> Result<Self>
    where
    State: Unfitted,
    Self: Sized;
}

/// Trait for prediction-capable models.
pub trait Predictor<T, X, State>: Model
where
T: Numeric,
X: Dataset<T>,
{
    /// Predicts outputs for input data.
    fn predict(&self, x: &X) -> Result<Vec<T>>
    where
    State: Fitted;
}
