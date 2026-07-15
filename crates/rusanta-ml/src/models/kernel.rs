// rusanta-ml/src/models/kernel.rs
// Kernel feature mapping — explicit feature space expansion
// Linear, Polynomial, RBF (Gaussian)

use crate::dataset::Dataset;

/// Supported kernels
#[derive(Debug, Clone)]
pub enum Kernel {
    Linear,
    Polynomial { degree: usize, coef0: f64 },
    RBF { gamma: f64 },
}

/// Kernel feature mapper
///
/// Transforms input features into higher-dimensional space.
/// This is an explicit mapping (NOT kernel trick).
#[derive(Debug, Clone)]
pub struct KernelMapper {
    pub kernel: Kernel,
}

impl KernelMapper {
    pub fn new(kernel: Kernel) -> Self {
        Self { kernel }
    }

    fn linear(x: &[f64]) -> Vec<f64> {
        x.to_vec()
    }

    fn polynomial(x: &[f64], degree: usize, coef0: f64) -> Vec<f64> {
        let mut feats = Vec::new();
        let d = x.len();

        // naive polynomial expansion (up to exact degree)
        fn expand(
            x: &[f64],
            degree: usize,
            coef0: f64,
            start: usize,
            current: f64,
            feats: &mut Vec<f64>,
        ) {
            if degree == 0 {
                feats.push(current);
                return;
            }
            for i in start..x.len() {
                expand(
                    x,
                    degree - 1,
                    coef0,
                    i,
                    current * (x[i] + coef0),
                       feats,
                );
            }
        }

        expand(x, degree, coef0, 0, 1.0, &mut feats);
        feats
    }

    fn rbf(x: &[f64], centers: &[Vec<f64>], gamma: f64) -> Vec<f64> {
        centers
        .iter()
        .map(|c| {
            x.iter()
            .zip(c.iter())
            .map(|(xi, ci)| (xi - ci).powi(2))
            .sum::<f64>()
            .mul_add(-gamma, 0.0)
            .exp()
        })
        .collect()
    }
}

impl<D> KernelMapper
where
D: Dataset<Feature = f64, Target = f64>,
{
    pub fn transform(&self, data: &D) -> Vec<Vec<f64>> {
        match &self.kernel {
            Kernel::Linear => (0..data.len())
            .map(|i| Self::linear(data.feature_row(i)))
            .collect(),

            Kernel::Polynomial { degree, coef0 } => (0..data.len())
            .map(|i| Self::polynomial(data.feature_row(i), *degree, *coef0))
            .collect(),

            Kernel::RBF { gamma } => {
                // use dataset itself as centers (Nyström-like)
                let centers: Vec<Vec<f64>> =
                (0..data.len()).map(|i| data.feature_row(i).to_vec()).collect();

                (0..data.len())
                .map(|i| Self::rbf(data.feature_row(i), &centers, *gamma))
                .collect()
            }
        }
    }
}
