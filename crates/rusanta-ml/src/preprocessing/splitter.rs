// rusanta-ml/src/preprocessing/splitter.rs

use crate::dataset::{Dataset, InMemoryDataset};

/// Dataset splitting utilities.
///
/// Provides train/test and train/validation/test splits,
/// similar to sklearn's `train_test_split`.
#[derive(Debug, Clone)]
pub struct Splitter {
    pub test_size: f64,
    pub shuffle: bool,
    pub seed: Option<u64>,
}

impl Default for Splitter {
    fn default() -> Self {
        Self {
            test_size: 0.2,
            shuffle: true,
            seed: None,
        }
    }
}

impl Splitter {
    /// Create a new splitter.
    pub fn new(test_size: f64) -> Self {
        assert!(
            test_size > 0.0 && test_size < 1.0,
            "test_size must be in (0, 1)"
        );
        Self {
            test_size,
            ..Default::default()
        }
    }

    /// Enable or disable shuffling.
    pub fn shuffle(mut self, enabled: bool) -> Self {
        self.shuffle = enabled;
        self
    }

    /// Set RNG seed (for reproducibility).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Split dataset into (train, test).
    pub fn split<D>(
        &self,
        data: &D,
    ) -> (
        InMemoryDataset<D::Feature, D::Target>,
        InMemoryDataset<D::Feature, D::Target>,
    )
    where
    D: Dataset,
    D::Feature: Copy,
    D::Target: Copy,
    {
        let n = data.len();
        let test_len = (n as f64 * self.test_size).round() as usize;
        let train_len = n - test_len;

        let mut indices: Vec<usize> = (0..n).collect();

        if self.shuffle {
            match self.seed {
                Some(seed) => {
                    use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
                    let mut rng = StdRng::seed_from_u64(seed);
                    indices.shuffle(&mut rng);
                }
                None => {
                    use rand::seq::SliceRandom;
                    let mut rng = rand::thread_rng();
                    indices.shuffle(&mut rng);
                }
            }
        }

        let (train_idx, test_idx) = indices.split_at(train_len);

        let build = |idxs: &[usize]| {
            let mut features = Vec::with_capacity(idxs.len());
            let mut targets = Vec::with_capacity(idxs.len());

            for &i in idxs {
                features.push(data.feature_row(i).to_vec());
                targets.push(data.get_target(i));
            }

            InMemoryDataset::new(features, targets)
        };

        (build(train_idx), build(test_idx))
    }

    /// Split dataset into (train, validation, test).
    ///
    /// Validation size is relative to the remaining data after test split.
    pub fn split_three<D>(
        &self,
        data: &D,
        val_size: f64,
    ) -> (
        InMemoryDataset<D::Feature, D::Target>,
        InMemoryDataset<D::Feature, D::Target>,
        InMemoryDataset<D::Feature, D::Target>,
    )
    where
    D: Dataset,
    D::Feature: Copy,
    D::Target: Copy,
    {
        assert!(val_size > 0.0 && val_size < 1.0);

        let (train_full, test) = self.split(data);

        let val_splitter = Splitter {
            test_size: val_size,
            shuffle: self.shuffle,
            seed: self.seed,
        };

        let (train, val) = val_splitter.split(&train_full);

        (train, val, test)
    }
}
