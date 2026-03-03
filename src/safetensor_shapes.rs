//! Shape-only view of a safetensors file (no weight data loaded).
//!
//! Used during model tracing to discover weight shapes from a checkpoint header
//! without reading the full tensor data into memory.

use std::collections::HashMap;

use safetensors::SafeTensors;

use crate::{dtype::DType, error::Error, module_api::ShapeProvider, shape::Shape};

/// Maps canonical weight names to their shapes, parsed from a safetensors header.
#[derive(Debug, Clone)]
pub struct SafeTensorShapes {
    map: HashMap<String, Shape>,
}

impl SafeTensorShapes {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| Error::RuntimeError(format!("parse safetensors: {}", e)))?;

        let mut map = HashMap::new();
        for name in st.names() {
            let tv = st
                .tensor(name)
                .map_err(|e| Error::RuntimeError(format!("tensor {}: {}", name, e)))?;
            let dims: Vec<i64> = tv.shape().iter().map(|&d| d as i64).collect();

            // BF16/F16 weights are promoted to F32 at load time because the
            // upstream `pjrt` crate lacks bf16/f16 host-buffer support.
            // When that's fixed, this can preserve the original dtype.
            let dtype = match tv.dtype() {
                safetensors::Dtype::F32
                | safetensors::Dtype::F16
                | safetensors::Dtype::BF16 => DType::F32,
                other => {
                    return Err(Error::UnsupportedDType {
                        key: name.to_string(),
                        dtype: format!("{:?}", other),
                    });
                }
            };

            // Normalize separators: HuggingFace uses dots, fusebox uses slashes.
            let canonical = name.replace('.', "/");
            map.insert(canonical, Shape::new(dims, dtype));
        }

        Ok(Self { map })
    }
}

impl ShapeProvider for SafeTensorShapes {
    fn shape_of(&self, full_name: &str) -> Result<Option<Shape>, Error> {
        Ok(self.map.get(full_name).cloned())
    }

    fn has_prefix(&self, prefix: &str) -> bool {
        let search = if prefix.ends_with('/') {
            prefix.to_string()
        } else {
            format!("{}/", prefix)
        };
        self.map.keys().any(|k| k.starts_with(&search))
    }
}
