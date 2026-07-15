use std::fmt;
use crate::error::Result;

/// A single processing step in a pipeline.
///
/// A step consumes data and produces data.
/// It may fail.
pub trait Step<D>: Send + Sync {
    /// Human-readable name of the step.
    fn name(&self) -> &str;

    /// Apply the transformation.
    fn run(&self, input: D) -> Result<D>;
}

/// A sequential data processing pipeline.
///
/// Pipeline owns its steps and executes them in order.
pub struct Pipeline<D> {
    steps: Vec<Box<dyn Step<D>>>,
}

impl<D> Pipeline<D> {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a step to the pipeline.
    pub fn add_step<S>(mut self, step: S) -> Self
    where
    S: Step<D> + 'static,
    {
        self.steps.push(Box::new(step));
        self
    }

    /// Execute the pipeline.
    ///
    /// Data flows through each step sequentially.
    pub fn run(&self, mut data: D) -> Result<D> {
        for step in &self.steps {
            data = step.run(data)?;
        }
        Ok(data)
    }

    /// Number of steps in the pipeline.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Inspect pipeline step names.
    pub fn steps(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.name()).collect()
    }
}

impl<D> Default for Pipeline<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D> fmt::Debug for Pipeline<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipeline")
        .field("steps", &self.steps())
        .finish()
    }
}

/* ===========================
 * Helper step implementations
 * =========================== */

/// A pipeline step created from a closure.
///
/// Useful for quick inline transformations.
pub struct FnStep<D, F>
where
F: Fn(D) -> Result<D> + Send + Sync,
{
    name: String,
    func: F,
}

impl<D, F> FnStep<D, F>
where
F: Fn(D) -> Result<D> + Send + Sync,
{
    pub fn new<N: Into<String>>(name: N, func: F) -> Self {
        Self {
            name: name.into(),
            func,
        }
    }
}

impl<D, F> Step<D> for FnStep<D, F>
where
F: Fn(D) -> Result<D> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, input: D) -> Result<D> {
        (self.func)(input)
    }
}
