use std::{fs, path::Path, sync::Arc};

use crate::{
    safetensor_shapes::SafeTensorShapes,
    signature::Signature,
    weights::WeightsF32,
};

#[derive(Clone)]
pub struct Checkpoint {
    bytes: Arc<Vec<u8>>,
    shapes: SafeTensorShapes,
}

impl Checkpoint {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|e| format!("read {:?}: {}", path, e))?;
        Self::from_bytes(bytes)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, String> {
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

    pub fn weights_f32_for_signature(&self, sig: &Signature) -> Result<WeightsF32, String> {
        WeightsF32::from_safetensors_for_weights_bytes(self.bytes(), sig)
    }
}