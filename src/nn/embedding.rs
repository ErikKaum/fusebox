use fusebox_macros::Module;

use crate::error::Error;
use crate::tensor::Tensor;

#[derive(Module)]
pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    pub fn forward(&self, indices: &Tensor) -> Result<Tensor, Error> {
        self.weight.gather(indices)
    }
}
