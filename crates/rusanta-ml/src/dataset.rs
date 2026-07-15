// rusanta-ml/src/dataset.rs

use std::fmt::Debug;

/// Core dataset trait for rusanta-ml.
///
/// This is intentionally minimal and **model-agnostic**.
/// It represents a supervised or unsupervised dataset in memory.
///
/// Design goals:
/// - no assumption about storage backend
/// - works with arrays, frames, or custom loaders
/// - zero-copy where possible
pub trait Dataset: Debug {
    /// Feature type (usually f32 / f64).
    type Feature: Copy + Debug;

    /// Target type (can be Feature, usize, bool, etc.).
    type Target: Copy + Debug;

    /// Number of samples (rows).
    fn len(&self) -> usize;

    /// Returns true if dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of features (columns).
    fn n_features(&self) -> usize;

    /// Get a single feature value.
    ///
    /// # Arguments
    /// - `row`: sample index
    /// - `col`: feature index
    fn get_feature(&self, row: usize, col: usize) -> Self::Feature;

    /// Get a full feature row.
    fn feature_row(&self, row: usize) -> &[Self::Feature];

    /// Get target value for a row.
    fn get_target(&self, row: usize) -> Self::Target;
}

/// Owned in-memory dataset.
///
/// This is the **default concrete dataset** used by most models.
#[derive(Debug, Clone)]
pub struct InMemoryDataset<F, T> {
    features: Vec<Vec<F>>,
    targets: Vec<T>,
}

impl<F, T> InMemoryDataset<F, T>
where
F: Copy + Debug,
T: Copy + Debug,
{
    /// Create a new dataset.
    ///
    /// # Panics
    /// - if features and targets length mismatch
    /// - if features are ragged
    pub fn new(features: Vec<Vec<F>>, targets: Vec<T>) -> Self {
        assert!(
            features.len() == targets.len(),
                "features and targets must have same length"
        );

        if !features.is_empty() {
            let cols = features[0].len();
            for row in &features {
                assert!(
                    row.len() == cols,
                        "all feature rows must have equal length"
                );
            }
        }

        Self { features, targets }
    }

    /// Borrow features matrix.
    pub fn features(&self) -> &[Vec<F>] {
        &self.features
    }

    /// Borrow targets.
    pub fn targets(&self) -> &[T] {
        &self.targets
    }
}

impl<F, T> Dataset for InMemoryDataset<F, T>
where
F: Copy + Debug,
T: Copy + Debug,
{
    type Feature = F;
    type Target = T;

    fn len(&self) -> usize {
        self.features.len()
    }

    fn n_features(&self) -> usize {
        if self.features.is_empty() {
            0
        } else {
            self.features[0].len()
        }
    }

    fn get_feature(&self, row: usize, col: usize) -> F {
        self.features[row][col]
    }

    fn feature_row(&self, row: usize) -> &[F] {
        &self.features[row]
    }

    fn get_target(&self, row: usize) -> T {
        self.targets[row]
    }
}
