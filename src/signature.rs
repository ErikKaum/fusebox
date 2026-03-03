use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::dtype::DType;
use crate::error::Error;
use crate::ir::{Function, ParamKind};
use crate::shape::Shape;
use crate::value::ValueId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamSpec {
    pub name: String,
    pub shape: Shape,
    pub value: ValueId,
    pub kind: ParamKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    pub fn input_params(&self) -> impl Iterator<Item = &ParamSpec> {
        self.params.iter().filter(|p| p.kind == ParamKind::Input)
    }

    pub fn index_of_name(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    pub fn spec(&self, idx: usize) -> &ParamSpec {
        &self.params[idx]
    }
}

#[derive(Debug, Clone)]
pub enum ParamData {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl ParamData {
    pub fn dtype(&self) -> DType {
        match self {
            ParamData::F32(_) => DType::F32,
            ParamData::I32(_) => DType::I32,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ParamData::F32(v) => v.len(),
            ParamData::I32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone)]
pub struct Inputs {
    sig: Arc<Signature>,
    values: Vec<Option<ParamData>>,
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
        let idx = self.resolve_name(name)?;
        self.set_data(idx, ParamData::F32(data))
    }

    pub fn set_i32(&mut self, name: &str, data: Vec<i32>) -> Result<(), Error> {
        let idx = self.resolve_name(name)?;
        self.set_data(idx, ParamData::I32(data))
    }

    pub fn set_input(&mut self, name: &str, data: Vec<f32>) -> Result<(), Error> {
        let idx = self.resolve_name(name)?;
        if self.sig.spec(idx).kind == ParamKind::Weight {
            return Err(Error::InvalidParam {
                msg: format!("cannot set weight {:?} via set_input()", name),
            });
        }
        self.set_data(idx, ParamData::F32(data))
    }

    pub fn set_input_i32(&mut self, name: &str, data: Vec<i32>) -> Result<(), Error> {
        let idx = self.resolve_name(name)?;
        if self.sig.spec(idx).kind == ParamKind::Weight {
            return Err(Error::InvalidParam {
                msg: format!("cannot set weight {:?} via set_input_i32()", name),
            });
        }
        self.set_data(idx, ParamData::I32(data))
    }

    fn resolve_name(&self, name: &str) -> Result<usize, Error> {
        self.sig
            .index_of_name(name)
            .ok_or_else(|| Error::InvalidParam {
                msg: format!("unknown parameter {:?}", name),
            })
    }

    fn set_data(&mut self, idx: usize, data: ParamData) -> Result<(), Error> {
        let spec = self.sig.spec(idx);

        let expected_dtype = spec.shape.dtype;
        let got_dtype = data.dtype();
        if expected_dtype != got_dtype {
            return Err(Error::InvalidParam {
                msg: format!(
                    "param {:?} expects dtype {}, got {}",
                    spec.name, expected_dtype, got_dtype
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

    pub fn into_ordered(self) -> Result<Vec<(Shape, ParamData)>, Error> {
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
