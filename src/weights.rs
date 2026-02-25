use std::collections::HashMap;

use safetensors::{Dtype as SdType, SafeTensors};

use crate::ir::ParamKind;
use crate::signature::Signature;

#[derive(Debug, Clone)]
pub struct WeightsF32 {
    data: HashMap<String, Vec<f32>>,
}

impl WeightsF32 {
    pub fn from_safetensors_for_weights_bytes(
        bytes: &[u8],
        sig: &Signature,
    ) -> Result<Self, String> {
        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| format!("parse safetensors: {}", e))?;
        Self::from_safetensors_inner(&st, sig)
    }

    pub fn apply_ref(&self, bindings: &mut crate::signature::InputsF32) -> Result<(), String> {
        for (name, data) in &self.data {
            bindings.set(name, data.clone())?;
        }
        Ok(())
    }

    fn from_safetensors_inner(st: &SafeTensors, sig: &Signature) -> Result<Self, String> {
        let mut out = HashMap::new();

        for p in sig.params() {
            if p.kind != ParamKind::Weight {
                continue;
            }

            let tv = st
                .tensor(&p.name)
                .map_err(|e| format!("missing weight {:?} in safetensors: {}", p.name, e))?;

            if tv.dtype() != SdType::F32 {
                return Err(format!(
                    "weight {:?} has dtype {:?}, expected F32",
                    p.name,
                    tv.dtype()
                ));
            }

            let file_shape: Vec<i64> = tv.shape().iter().map(|&d| d as i64).collect();
            if file_shape != p.shape.dims {
                return Err(format!(
                    "weight {:?} shape mismatch: file={:?}, expected={:?}",
                    p.name, file_shape, p.shape.dims
                ));
            }

            let raw = tv.data();
            let vals =
                bytes_to_f32_le(raw).map_err(|e| format!("weight {:?}: {}", p.name, e))?;

            let want = p.shape.dims.iter().map(|&d| d as i128).product::<i128>();
            if vals.len() as i128 != want {
                return Err(format!(
                    "weight {:?} element count mismatch: got={}, expected={}",
                    p.name,
                    vals.len(),
                    want
                ));
            }

            out.insert(p.name.clone(), vals);
        }

        Ok(Self { data: out })
    }
}

fn bytes_to_f32_le(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 4 != 0 {
        return Err(format!("byte length {} not divisible by 4", bytes.len()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}
