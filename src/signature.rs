// src/signature.rs
use std::collections::HashMap;
use std::sync::Arc;

use crate::dtype::DType;
use crate::ir::{Function, ParamKind};
use crate::shape::Shape;
use crate::tensor::Tensor;
use crate::value::ValueId;

#[derive(Debug, Clone)]
pub struct ParamSpec {
    pub name: String,
    pub shape: Shape,
    pub value: ValueId,
    pub kind: ParamKind,
}

#[derive(Debug, Clone)]
pub struct Signature {
    params: Vec<ParamSpec>,            // in parameter order
    by_name: HashMap<String, usize>,   // name -> index
    by_value: HashMap<ValueId, usize>, // ValueId -> index (only params)
}

impl Signature {
    pub fn from_function(f: &Function) -> Self {
        let mut params = Vec::with_capacity(f.params.len());
        let mut by_name = HashMap::new();
        let mut by_value = HashMap::new();

        for (i, p) in f.params.iter().enumerate() {
            let spec = ParamSpec {
                name: p.name.clone(),
                shape: p.shape.clone(),
                value: p.value,
                kind: p.kind,
            };
            by_name.insert(spec.name.clone(), i);
            by_value.insert(spec.value, i);
            params.push(spec);
        }

        Self {
            params,
            by_name,
            by_value,
        }
    }

    pub fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    pub fn weight_params(&self) -> impl Iterator<Item = &ParamSpec> {
        self.params.iter().filter(|p| p.kind == ParamKind::Weight)
    }

    pub fn input_params(&self) -> impl Iterator<Item = &ParamSpec> {
        self.params.iter().filter(|p| p.kind == ParamKind::Input)
    }

    pub fn index_of_name(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    pub fn index_of_value(&self, v: ValueId) -> Option<usize> {
        self.by_value.get(&v).copied()
    }

    pub fn spec(&self, idx: usize) -> &ParamSpec {
        &self.params[idx]
    }
}

#[derive(Clone)]
pub struct InputsF32 {
    sig: Arc<Signature>,
    // one entry per param, same order
    values: Vec<Option<Vec<f32>>>,
}

impl InputsF32 {
    pub fn new(sig: Arc<Signature>) -> Self {
        let n = sig.params().len();
        Self {
            sig,
            values: vec![None; n],
        }
    }

    pub fn signature(&self) -> &Signature {
        &self.sig
    }

    pub fn set(&mut self, name: &str, data: Vec<f32>) -> Result<(), String> {
        let idx = self
            .sig
            .index_of_name(name)
            .ok_or_else(|| format!("unknown input name {:?}", name))?;
        self.set_idx(idx, data)
    }

    pub fn set_tensor(&mut self, t: &Tensor, data: Vec<f32>) -> Result<(), String> {
        let idx = self
            .sig
            .index_of_value(t.value)
            .ok_or_else(|| format!("tensor {} is not a function parameter", t.value))?;
        self.set_idx(idx, data)
    }

    fn set_idx(&mut self, idx: usize, data: Vec<f32>) -> Result<(), String> {
        let spec = self.sig.spec(idx);

        if spec.shape.dtype != DType::F32 {
            return Err(format!(
                "only f32 supported for now; param {:?} has dtype {}",
                spec.name, spec.shape.dtype
            ));
        }

        let want = numel(&spec.shape)?;
        if data.len() as i128 != want {
            return Err(format!(
                "param {:?} expects shape {:?} ({} elements), got {}",
                spec.name,
                spec.shape.dims,
                want,
                data.len()
            ));
        }

        self.values[idx] = Some(data);
        Ok(())
    }

    pub fn into_ordered(self) -> Result<Vec<(Shape, Vec<f32>)>, String> {
        let mut out = Vec::with_capacity(self.values.len());
        for (i, v) in self.values.into_iter().enumerate() {
            let spec = self.sig.spec(i);
            let data = v.ok_or_else(|| format!("missing value for param {:?}", spec.name))?;
            out.push((spec.shape.clone(), data));
        }
        Ok(out)
    }
}

fn numel(shape: &Shape) -> Result<i128, String> {
    // For now we require all dims to be >= 1 (no dynamic '?' and no zero dims).
    // You can relax this later.
    let mut prod: i128 = 1;
    for &d in &shape.dims {
        if d <= 0 {
            return Err(format!("unsupported dim {} in shape {:?}", d, shape.dims));
        }
        prod *= d as i128;
    }
    Ok(prod)
}
