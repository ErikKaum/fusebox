use fusebox_macros::Module;

use crate::error::Error;
use crate::tensor::Tensor;

/// Embedding table lookup: maps integer token ids to dense vectors.
#[derive(Module)]
pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    pub fn forward(&self, indices: &Tensor) -> Result<Tensor, Error> {
        self.weight.gather(indices)
    }
}
