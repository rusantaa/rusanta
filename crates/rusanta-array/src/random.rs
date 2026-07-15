use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rusanta_core::{Numeric, Result};
use crate::ndarray::NdArray;

/// Random number generator wrapper for Rusanta.
///
/// This avoids global RNG state and allows reproducibility.
#[derive(Debug, Clone)]
pub struct Random {
    rng: StdRng,
}

impl Random {
    /// Creates a new RNG with entropy from the OS.
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
        }
    }

    /// Creates a new RNG with a fixed seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /* ===========================
     * Random array generators
     * =========================== */

    /// Generates an array with values sampled uniformly from [0, 1).
    pub fn uniform<T>(&mut self, shape: &[usize]) -> Result<NdArray<T>>
    where
    T: Numeric + rand::distributions::uniform::SampleUniform,
    {
        self.uniform_range(T::zero(), T::one(), shape)
    }

    /// Generates an array with values sampled uniformly from [low, high).
    pub fn uniform_range<T>(
        &mut self,
        low: T,
        high: T,
        shape: &[usize],
    ) -> Result<NdArray<T>>
    where
    T: Numeric + rand::distributions::uniform::SampleUniform,
    {
        let size: usize = shape.iter().product();
        let dist = Uniform::new(low, high);

        let mut buf = Vec::with_capacity(size);
        for _ in 0..size {
            buf.push(dist.sample(&mut self.rng));
        }

        NdArray::new(buf.into(), shape.to_vec())
    }

    /// Generates an array of random integers in [low, high).
    pub fn randint<T>(
        &mut self,
        low: T,
        high: T,
        shape: &[usize],
    ) -> Result<NdArray<T>>
    where
    T: Numeric + rand::distributions::uniform::SampleUniform,
    {
        self.uniform_range(low, high, shape)
    }

    /// Generates an array with values sampled from a normal distribution.
    pub fn normal<T>(
        &mut self,
        mean: T,
        std: T,
        shape: &[usize],
    ) -> Result<NdArray<T>>
    where
    T: Numeric
    + rand_distr::num_traits::Float
    + rand_distr::Distribution<T>,
    {
        let size: usize = shape.iter().product();
        let dist = rand_distr::Normal::new(mean, std)
        .map_err(|_| rusanta_core::RusantaError::InvalidValue {
            message: "invalid normal distribution parameters".into(),
        })?;

        let mut buf = Vec::with_capacity(size);
        for _ in 0..size {
            buf.push(dist.sample(&mut self.rng));
        }

        NdArray::new(buf.into(), shape.to_vec())
    }
}

/* ===========================
 * Convenience free functions
 * =========================== */

/// Uniform random array with OS entropy.
pub fn rand_uniform<T>(shape: &[usize]) -> Result<NdArray<T>>
where
T: Numeric + rand::distributions::uniform::SampleUniform,
{
    Random::new().uniform(shape)
}

/// Normal random array with OS entropy.
pub fn rand_normal<T>(mean: T, std: T, shape: &[usize]) -> Result<NdArray<T>>
where
T: Numeric
+ rand_distr::num_traits::Float
+ rand_distr::Distribution<T>,
{
    Random::new().normal(mean, std, shape)
}

/// Random integer array with OS entropy.
pub fn rand_int<T>(low: T, high: T, shape: &[usize]) -> Result<NdArray<T>>
where
T: Numeric + rand::distributions::uniform::SampleUniform,
{
    Random::new().randint(low, high, shape)
}
