use std::collections::HashMap;
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::Error;
use crate::ir::{Function, ParamKind};
use crate::shape::Shape;
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
    params: Vec<ParamSpec>,
    by_name: HashMap<String, usize>,
}

impl Signature {
    pub fn from_function(f: &Function) -> Self {
        let mut params = Vec::with_capacity(f.params.len());
        let mut by_name = HashMap::new();

        for (i, p) in f.params.iter().enumerate() {
            let spec = ParamSpec {
                name: p.name.clone(),
                shape: p.shape.clone(),
                value: p.value,
                kind: p.kind,
            };
            by_name.insert(spec.name.clone(), i);
            params.push(spec);
        }

        Self { params, by_name }
    }

    pub fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    pub fn weight_params(&self) -> impl Iterator<Item = &ParamSpec> {
        self.params.iter().filter(|p| p.kind == ParamKind::Weight)
    }

    pub fn index_of_name(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    pub fn spec(&self, idx: usize) -> &ParamSpec {
        &self.params[idx]
    }
}

#[derive(Clone)]
pub struct Inputs {
    sig: Arc<Signature>,
    values: Vec<Option<Vec<f32>>>,
}

impl Inputs {
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

    pub fn set(&mut self, name: &str, data: Vec<f32>) -> Result<(), Error> {
        let idx = self
            .sig
            .index_of_name(name)
            .ok_or_else(|| Error::InvalidParam {
                msg: format!("unknown parameter {:?}", name),
            })?;
        self.set_idx(idx, data)
    }

    pub fn set_input(&mut self, name: &str, data: Vec<f32>) -> Result<(), Error> {
        let idx = self
            .sig
            .index_of_name(name)
            .ok_or_else(|| Error::InvalidParam {
                msg: format!("unknown input {:?}", name),
            })?;
        if self.sig.spec(idx).kind == ParamKind::Weight {
            return Err(Error::InvalidParam {
                msg: format!("cannot set weight {:?} via set_input()", name),
            });
        }
        self.set(name, data)
    }

    fn set_idx(&mut self, idx: usize, data: Vec<f32>) -> Result<(), Error> {
        let spec = self.sig.spec(idx);

        if spec.shape.dtype != DType::F32 {
            return Err(Error::InvalidParam {
                msg: format!(
                    "only f32 supported for now; param {:?} has dtype {}",
                    spec.name, spec.shape.dtype
                ),
            });
        }

        let want = numel(&spec.shape)?;
        if data.len() as i128 != want {
            return Err(Error::InvalidParam {
                msg: format!(
                    "param {:?} expects shape {:?} ({} elements), got {}",
                    spec.name,
                    spec.shape.dims,
                    want,
                    data.len()
                ),
            });
        }

        self.values[idx] = Some(data);
        Ok(())
    }

    pub fn into_ordered(self) -> Result<Vec<(Shape, Vec<f32>)>, Error> {
        let sig = self.sig;
        let mut out = Vec::with_capacity(self.values.len());
        for (i, v) in self.values.into_iter().enumerate() {
            let spec = sig.spec(i);
            let data = v.ok_or_else(|| Error::InvalidParam {
                msg: format!("missing value for param {:?}", spec.name),
            })?;
            out.push((spec.shape.clone(), data));
        }
        Ok(out)
    }
}

fn numel(shape: &Shape) -> Result<i128, Error> {
    let mut prod: i128 = 1;
    for &d in &shape.dims {
        if d <= 0 {
            return Err(Error::InvalidParam {
                msg: format!("unsupported dim {} in shape {:?}", d, shape.dims),
            });
        }
        prod *= d as i128;
    }
    Ok(prod)
}
