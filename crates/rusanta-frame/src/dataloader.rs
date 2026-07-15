use crate::error::Result;
use std::sync::Arc;

/// A dataset that can be iterated by index.
///
/// This is intentionally minimal and framework-level.
pub trait IndexDataset<D>: Send + Sync {
    /// Total number of samples.
    fn len(&self) -> usize;

    /// Get a single sample by index.
    fn get(&self, index: usize) -> Result<D>;
}

/// DataLoader configuration.
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    pub batch_size: usize,
    pub shuffle: bool,
    pub drop_last: bool,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: false,
            drop_last: false,
        }
    }
}

/// A batched data loader.
///
/// Inspired by PyTorch's DataLoader, but simpler and synchronous.
pub struct DataLoader<D> {
    dataset: Arc<dyn IndexDataset<D>>,
    indices: Vec<usize>,
    config: DataLoaderConfig,
    position: usize,
}

impl<D> DataLoader<D> {
    /// Create a new DataLoader.
    pub fn new(
        dataset: Arc<dyn IndexDataset<D>>,
        config: DataLoaderConfig,
    ) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();

        if config.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        Self {
            dataset,
            indices,
            config,
            position: 0,
        }
    }

    /// Reset loader for a new epoch.
    pub fn reset(&mut self) {
        self.position = 0;

        if self.config.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Number of batches per epoch.
    pub fn num_batches(&self) -> usize {
        let total = self.indices.len();
        let bs = self.config.batch_size;

        if self.config.drop_last {
            total / bs
        } else {
            (total + bs - 1) / bs
        }
    }
}

impl<D> Iterator for DataLoader<D> {
    type Item = Result<Vec<D>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.indices.len() {
            return None;
        }

        let end = (self.position + self.config.batch_size)
        .min(self.indices.len());

        if self.config.drop_last && end - self.position < self.config.batch_size {
            self.position = self.indices.len();
            return None;
        }

        let mut batch = Vec::with_capacity(end - self.position);
        for &idx in &self.indices[self.position..end] {
            match self.dataset.get(idx) {
                Ok(item) => batch.push(item),
                Err(e) => return Some(Err(e)),
            }
        }

        self.position = end;
        Some(Ok(batch))
    }
}
