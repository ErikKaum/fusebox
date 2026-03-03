use std::collections::HashMap;

use safetensors::SafeTensors;

use crate::{dtype::DType, error::Error, module_api::ShapeProvider, shape::Shape};

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

            let dtype = match tv.dtype() {
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::F16 => DType::F16,
                safetensors::Dtype::BF16 => DType::BF16,
                other => {
                    return Err(Error::UnsupportedDType {
                        key: name.to_string(),
                        dtype: format!("{:?}", other),
                    });
                }
            };

            map.insert(name.to_string(), Shape::new(dims, dtype));
        }

        Ok(Self { map })
    }
}

impl ShapeProvider for SafeTensorShapes {
    fn shape_of(&self, full_name: &str) -> Result<Option<Shape>, Error> {
        Ok(self.map.get(full_name).cloned())
    }
}
