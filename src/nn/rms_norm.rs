use fusebox_macros::Module;

use crate::error::Error;
use crate::tensor::Tensor;

/// Root Mean Square Layer Normalization (used in LLaMA-family models).
#[derive(Module)]
pub struct RmsNorm {
    pub weight: Tensor,
}

impl RmsNorm {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        self.forward_with_eps(x, 1e-5)
    }

    pub fn forward_with_eps(&self, x: &Tensor, eps: f64) -> Result<Tensor, Error> {
        let x_sq = x.mul(x)?;
        let variance = x_sq.mean_keepdim(&[-1])?;
        let eps = variance.full_like(eps);
        let rms = variance.add(&eps)?.rsqrt();
        x.mul(&rms)?.mul(&self.weight)
    }
}
