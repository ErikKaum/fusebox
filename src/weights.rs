use std::collections::HashMap;

use safetensors::{Dtype as SdType, SafeTensors};

use crate::error::Error;
use crate::ir::ParamKind;
use crate::signature::{Inputs, Signature};

#[derive(Debug, Clone)]
pub struct Weights {
    data: HashMap<String, Vec<f32>>,
}

impl Weights {
    pub fn from_safetensors_bytes(bytes: &[u8], sig: &Signature) -> Result<Self, Error> {
        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| Error::RuntimeError(format!("parse safetensors: {}", e)))?;
        Self::from_safetensors_inner(&st, sig)
    }

    pub fn apply_ref(&self, bindings: &mut Inputs) -> Result<(), Error> {
        for (name, data) in &self.data {
            bindings.set(name, data.clone())?;
        }
        Ok(())
    }

    fn from_safetensors_inner(st: &SafeTensors, sig: &Signature) -> Result<Self, Error> {
        let mut out = HashMap::new();

        for p in sig.params() {
            if p.kind != ParamKind::Weight {
                continue;
            }

            // Try the canonical slash-based name first, then fall back to
            // dot-separated keys (raw HuggingFace convention).
            let tv = st.tensor(&p.name).or_else(|_| {
                let dot_name = p.name.replace('/', ".");
                st.tensor(&dot_name)
            }).map_err(|e| {
                Error::RuntimeError(format!("missing weight {:?} in safetensors: {}", p.name, e))
            })?;

            let file_shape: Vec<i64> = tv.shape().iter().map(|&d| d as i64).collect();
            if file_shape != p.shape.dims {
                return Err(Error::RuntimeError(format!(
                    "weight {:?} shape mismatch: file={:?}, expected={:?}",
                    p.name, file_shape, p.shape.dims
                )));
            }

            let raw = tv.data();
            let vals = match tv.dtype() {
                SdType::F32 => bytes_to_f32(raw)?,
                SdType::BF16 => bf16_bytes_to_f32(raw)?,
                SdType::F16 => f16_bytes_to_f32(raw)?,
                other => {
                    return Err(Error::UnsupportedDType {
                        key: p.name.clone(),
                        dtype: format!("{:?}", other),
                    });
                }
            };

            let want = p.shape.dims.iter().map(|&d| d as i128).product::<i128>();
            if vals.len() as i128 != want {
                return Err(Error::RuntimeError(format!(
                    "weight {:?} element count mismatch: got={}, expected={}",
                    p.name,
                    vals.len(),
                    want
                )));
            }

            out.insert(p.name.clone(), vals);
        }

        Ok(Self { data: out })
    }
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, Error> {
    if bytes.len() % 4 != 0 {
        return Err(Error::RuntimeError(format!(
            "f32 byte length {} not divisible by 4",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, Error> {
    if bytes.len() % 2 != 0 {
        return Err(Error::RuntimeError(format!(
            "bf16 byte length {} not divisible by 2",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        // bfloat16 is the upper 16 bits of IEEE 754 float32
        let bits = ((chunk[1] as u32) << 24) | ((chunk[0] as u32) << 16);
        out.push(f32::from_bits(bits));
    }
    Ok(out)
}

fn f16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, Error> {
    if bytes.len() % 2 != 0 {
        return Err(Error::RuntimeError(format!(
            "f16 byte length {} not divisible by 2",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(f16_to_f32(bits));
    }
    Ok(out)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormalized f16 → normalized f32
        let mut e = 0i32;
        let mut m = mant;
        while m & 0x400 == 0 {
            m <<= 1;
            e += 1;
        }
        let exp32 = (127 - 15 - e) as u32;
        let mant32 = (m & 0x3FF) << 13;
        return f32::from_bits((sign << 31) | (exp32 << 23) | mant32);
    }

    if exp == 31 {
        // Inf / NaN
        let mant32 = mant << 13;
        return f32::from_bits((sign << 31) | (0xFF << 23) | mant32);
    }

    let exp32 = exp + (127 - 15);
    let mant32 = mant << 13;
    f32::from_bits((sign << 31) | (exp32 << 23) | mant32)
}
