use fusebox_macros::Module;

use crate::error::Error;
use crate::tensor::Tensor;

/// Linear layer with weight stored in `[out_features, in_features]` layout
/// (PyTorch / HuggingFace convention). Computes `x @ W^T + bias`.
#[derive(Module)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let wt = self.weight.transpose(&[1, 0])?;
        let y = x.matmul(&wt)?;
        match &self.bias {
            Some(b) => y.add(b),
            None => Ok(y),
        }
    }
}
