// src/safetensors_shapes.rs
use std::{collections::HashMap, fs, path::Path};

use safetensors::{Dtype as SdType, SafeTensors};

use crate::dtype::DType;
use crate::error::Error;
use crate::module_api::ShapeProvider;
use crate::shape::Shape;

#[derive(Debug, Clone)]
pub struct SafeTensorShapes {
    // name -> (dtype, dims)
    map: HashMap<String, (SdType, Vec<i64>)>,
}

impl SafeTensorShapes {
    // TODO note that this reads the entire file now, future let's just peek what we need
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|e| format!("read {:?}: {}", path, e))?;

        let st = SafeTensors::deserialize(&bytes)
            .map_err(|e| format!("parse safetensors {:?}: {}", path, e))?;

        let mut map = HashMap::new();
        for name in st.names() {
            let tv = st
                .tensor(name)
                .map_err(|e| format!("read tensor {:?} from {:?}: {}", name, path, e))?;

            let dims: Vec<i64> = tv.shape().iter().map(|&d| d as i64).collect();
            map.insert(name.to_string(), (tv.dtype(), dims));
        }

        Ok(Self { map })
    }
}

impl ShapeProvider for SafeTensorShapes {
    fn shape_of(&self, full_name: &str) -> Result<Option<Shape>, Error> {
        let Some((sd, dims)) = self.map.get(full_name) else {
            return Ok(None);
        };

        let dtype = match sd {
            SdType::F32 => DType::F32,
            SdType::F16 => DType::F16,
            SdType::BF16 => DType::BF16,
            other => {
                return Err(Error::UnsupportedDType {
                    key: full_name.to_string(),
                    dtype: format!("{:?}", other),
                });
            }
        };

        Ok(Some(Shape::new(dims.clone(), dtype)))
    }
}
