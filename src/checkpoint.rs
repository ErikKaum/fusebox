use std::{fs, path::Path, sync::Arc};

use crate::{
    error::Error, safetensor_shapes::SafeTensorShapes, signature::Signature, weights::Weights,
};

#[derive(Clone)]
pub struct Checkpoint {
    bytes: Arc<Vec<u8>>,
    shapes: SafeTensorShapes,
}

impl Checkpoint {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let bytes =
            fs::read(path).map_err(|e| Error::RuntimeError(format!("read {:?}: {}", path, e)))?;
        Self::from_bytes(bytes)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, Error> {
        let bytes = Arc::new(bytes);
        let shapes = SafeTensorShapes::from_bytes(&bytes)?;
        Ok(Self { bytes, shapes })
    }

    pub fn shapes(&self) -> &SafeTensorShapes {
        &self.shapes
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn load_weights(&self, sig: &Signature) -> Result<Weights, Error> {
        Weights::from_safetensors_bytes(self.bytes(), sig)
    }
}
