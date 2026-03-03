use core::fmt;
use std::cell::RefCell;
use std::rc::Rc;

use crate::builder::FuncBuilder;
use crate::dtype::DType;
use crate::error::Error;
use crate::ir::CompareDirection;
use crate::shape::Shape;
use crate::value::ValueId;

/// A symbolic tensor: shape metadata + SSA value id + a reference to the
/// computation graph being built.
///
/// This is NOT a data buffer. It represents "the result of some operation"
/// inside a traced computation graph. Operations on tensors record new
/// instructions in the graph.
pub struct Tensor {
    pub shape: Shape,
    pub value: ValueId,
    pub(crate) graph: Rc<RefCell<FuncBuilder>>,
}

impl Tensor {
    pub(crate) fn new(shape: Shape, value: ValueId, graph: &Rc<RefCell<FuncBuilder>>) -> Self {
        Self {
            shape,
            value,
            graph: graph.clone(),
        }
    }

    pub fn dtype(&self) -> DType {
        self.shape.dtype
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    fn check_same_graph(&self, other: &Tensor) -> Result<(), Error> {
        if !Rc::ptr_eq(&self.graph, &other.graph) {
            return Err(Error::GraphMismatch);
        }
        Ok(())
    }

    // ── Matmul ──────────────────────────────────────────────────────

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, Error> {
        self.check_same_graph(other)?;
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.matmul(&self.shape, self.value, &other.shape, other.value)?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    // ── Binary ops (with broadcasting) ──────────────────────────────

    pub fn add(&self, other: &Tensor) -> Result<Tensor, Error> {
        self.check_same_graph(other)?;
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.add(&self.shape, self.value, &other.shape, other.value)?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, Error> {
        self.check_same_graph(other)?;
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.sub(&self.shape, self.value, &other.shape, other.value)?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, Error> {
        self.check_same_graph(other)?;
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.mul(&self.shape, self.value, &other.shape, other.value)?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor, Error> {
        self.check_same_graph(other)?;
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.div(&self.shape, self.value, &other.shape, other.value)?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    pub fn maximum(&self, other: &Tensor) -> Result<Tensor, Error> {
        self.check_same_graph(other)?;
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.maximum(&self.shape, self.value, &other.shape, other.value)?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    // ── Unary ops ───────────────────────────────────────────────────

    pub fn neg(&self) -> Tensor {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.neg(&self.shape, self.value);
        drop(b);
        Tensor::new(shape, value, &self.graph)
    }

    pub fn exp(&self) -> Tensor {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.exp(&self.shape, self.value);
        drop(b);
        Tensor::new(shape, value, &self.graph)
    }

    pub fn log(&self) -> Tensor {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.log(&self.shape, self.value);
        drop(b);
        Tensor::new(shape, value, &self.graph)
    }

    pub fn sqrt(&self) -> Tensor {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.sqrt(&self.shape, self.value);
        drop(b);
        Tensor::new(shape, value, &self.graph)
    }

    pub fn rsqrt(&self) -> Tensor {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.rsqrt(&self.shape, self.value);
        drop(b);
        Tensor::new(shape, value, &self.graph)
    }

    pub fn abs(&self) -> Tensor {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.abs(&self.shape, self.value);
        drop(b);
        Tensor::new(shape, value, &self.graph)
    }

    pub fn tanh(&self) -> Tensor {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.tanh(&self.shape, self.value);
        drop(b);
        Tensor::new(shape, value, &self.graph)
    }

    pub fn sigmoid(&self) -> Tensor {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.logistic(&self.shape, self.value);
        drop(b);
        Tensor::new(shape, value, &self.graph)
    }

    pub fn cos(&self) -> Tensor {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.cosine(&self.shape, self.value);
        drop(b);
        Tensor::new(shape, value, &self.graph)
    }

    pub fn sin(&self) -> Tensor {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.sine(&self.shape, self.value);
        drop(b);
        Tensor::new(shape, value, &self.graph)
    }

    pub fn to_dtype(&self, dtype: DType) -> Tensor {
        if self.shape.dtype == dtype {
            return self.clone();
        }
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.convert(&self.shape, self.value, dtype);
        drop(b);
        Tensor::new(shape, value, &self.graph)
    }

    // ── Activations ─────────────────────────────────────────────────

    pub fn relu(&self) -> Tensor {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.relu(&self.shape, self.value);
        drop(b);
        Tensor::new(shape, value, &self.graph)
    }

    pub fn silu(&self) -> Tensor {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.silu(&self.shape, self.value);
        drop(b);
        Tensor::new(shape, value, &self.graph)
    }

    pub fn gelu(&self) -> Tensor {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.gelu(&self.shape, self.value);
        drop(b);
        Tensor::new(shape, value, &self.graph)
    }

    // ── Shape manipulation ──────────────────────────────────────────

    pub fn reshape(&self, new_dims: &[i64]) -> Result<Tensor, Error> {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.reshape(&self.shape, self.value, new_dims)?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    pub fn transpose(&self, permutation: &[i64]) -> Result<Tensor, Error> {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.transpose(&self.shape, self.value, permutation)?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    pub fn unsqueeze(&self, axis: i64) -> Result<Tensor, Error> {
        let rank = self.rank() as i64;
        let a = if axis < 0 { axis + rank + 1 } else { axis } as usize;
        let mut new_dims = self.shape.dims.clone();
        new_dims.insert(a, 1);
        self.reshape(&new_dims)
    }

    pub fn squeeze(&self, axis: i64) -> Result<Tensor, Error> {
        let rank = self.rank() as i64;
        let a = if axis < 0 { axis + rank } else { axis } as usize;
        if self.shape.dims[a] != 1 {
            return Err(Error::InvalidParam {
                msg: format!(
                    "squeeze: dim {a} has size {}, expected 1",
                    self.shape.dims[a]
                ),
            });
        }
        let mut new_dims = self.shape.dims.clone();
        new_dims.remove(a);
        self.reshape(&new_dims)
    }

    pub fn expand(&self, target_dims: &[i64]) -> Result<Tensor, Error> {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.expand(&self.shape, self.value, target_dims)?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    // ── Reductions ──────────────────────────────────────────────────

    pub fn sum(&self, axes: &[i64]) -> Result<Tensor, Error> {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.reduce_sum(&self.shape, self.value, axes)?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    pub fn max(&self, axes: &[i64]) -> Result<Tensor, Error> {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.reduce_max(&self.shape, self.value, axes)?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    pub fn min(&self, axes: &[i64]) -> Result<Tensor, Error> {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.reduce_min(&self.shape, self.value, axes)?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    pub fn argmax(&self, axis: i64) -> Result<Tensor, Error> {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.argmax(&self.shape, self.value, axis)?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    pub fn mean(&self, axes: &[i64]) -> Result<Tensor, Error> {
        let count: i64 = axes
            .iter()
            .map(|&a| {
                let rank = self.rank() as i64;
                let idx = if a < 0 {
                    (a + rank) as usize
                } else {
                    a as usize
                };
                self.shape.dims[idx]
            })
            .product();

        let sum = self.sum(axes)?;
        let count_t = sum.full_like(count as f64);
        sum.div(&count_t)
    }

    // ── Softmax ─────────────────────────────────────────────────────

    pub fn softmax(&self, axis: i64) -> Result<Tensor, Error> {
        let max_val = self.max(&[axis])?;
        let max_val = max_val.unsqueeze(axis)?;
        let shifted = self.sub(&max_val)?;
        let exps = shifted.exp();
        let sum_val = exps.sum(&[axis])?;
        let sum_val = sum_val.unsqueeze(axis)?;
        exps.div(&sum_val)
    }

    // ── Constants ───────────────────────────────────────────────────

    pub fn full_like(&self, value: f64) -> Tensor {
        let mut b = self.graph.borrow_mut();
        let (shape, val) = b.constant(value, &self.shape);
        drop(b);
        Tensor::new(shape, val, &self.graph)
    }

    pub fn zeros_like(&self) -> Tensor {
        self.full_like(0.0)
    }

    pub fn ones_like(&self) -> Tensor {
        self.full_like(1.0)
    }

    // ── Scalar binary ops ───────────────────────────────────────────

    pub(crate) fn scalar_const(&self, value: f64) -> Tensor {
        let mut b = self.graph.borrow_mut();
        let scalar_shape = Shape::new(vec![], self.shape.dtype);
        let (shape, val) = b.constant(value, &scalar_shape);
        drop(b);
        Tensor::new(shape, val, &self.graph)
    }

    pub fn add_scalar(&self, value: f64) -> Result<Tensor, Error> {
        let s = self.scalar_const(value);
        self.add(&s)
    }

    pub fn sub_scalar(&self, value: f64) -> Result<Tensor, Error> {
        let s = self.scalar_const(value);
        self.sub(&s)
    }

    pub fn mul_scalar(&self, value: f64) -> Result<Tensor, Error> {
        let s = self.scalar_const(value);
        self.mul(&s)
    }

    pub fn div_scalar(&self, value: f64) -> Result<Tensor, Error> {
        let s = self.scalar_const(value);
        self.div(&s)
    }

    // ── Keepdim reductions ──────────────────────────────────────────

    pub fn sum_keepdim(&self, axes: &[i64]) -> Result<Tensor, Error> {
        let reduced = self.sum(axes)?;
        self.restore_reduced_dims(&reduced, axes)
    }

    pub fn max_keepdim(&self, axes: &[i64]) -> Result<Tensor, Error> {
        let reduced = self.max(axes)?;
        self.restore_reduced_dims(&reduced, axes)
    }

    pub fn min_keepdim(&self, axes: &[i64]) -> Result<Tensor, Error> {
        let reduced = self.min(axes)?;
        self.restore_reduced_dims(&reduced, axes)
    }

    pub fn mean_keepdim(&self, axes: &[i64]) -> Result<Tensor, Error> {
        let reduced = self.mean(axes)?;
        self.restore_reduced_dims(&reduced, axes)
    }

    fn restore_reduced_dims(&self, reduced: &Tensor, axes: &[i64]) -> Result<Tensor, Error> {
        let rank = self.rank() as i64;
        let mut normalized: Vec<usize> = axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (a + rank) as usize
                } else {
                    a as usize
                }
            })
            .collect();
        normalized.sort();

        let mut new_dims = reduced.shape.dims.clone();
        for &a in &normalized {
            new_dims.insert(a, 1);
        }
        reduced.reshape(&new_dims)
    }

    // ── Concatenate ─────────────────────────────────────────────────

    pub fn cat(tensors: &[&Tensor], axis: i64) -> Result<Tensor, Error> {
        if tensors.is_empty() {
            return Err(Error::InvalidParam {
                msg: "cat: need at least one tensor".into(),
            });
        }
        let graph = &tensors[0].graph;
        for t in tensors.iter().skip(1) {
            if !Rc::ptr_eq(graph, &t.graph) {
                return Err(Error::GraphMismatch);
            }
        }
        let shapes: Vec<&Shape> = tensors.iter().map(|t| &t.shape).collect();
        let vals: Vec<ValueId> = tensors.iter().map(|t| t.value).collect();
        let mut b = graph.borrow_mut();
        let (shape, value) = b.concatenate(&shapes, &vals, axis)?;
        drop(b);
        Ok(Tensor::new(shape, value, graph))
    }

    // ── Slice ───────────────────────────────────────────────────────

    pub fn slice_range(
        &self,
        start_indices: &[i64],
        limit_indices: &[i64],
    ) -> Result<Tensor, Error> {
        let strides = vec![1i64; self.rank()];
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.slice(
            &self.shape,
            self.value,
            start_indices,
            limit_indices,
            &strides,
        )?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    pub fn slice_with_strides(
        &self,
        start_indices: &[i64],
        limit_indices: &[i64],
        strides: &[i64],
    ) -> Result<Tensor, Error> {
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.slice(
            &self.shape,
            self.value,
            start_indices,
            limit_indices,
            strides,
        )?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    pub fn narrow(&self, axis: i64, start: i64, len: i64) -> Result<Tensor, Error> {
        let rank = self.rank() as i64;
        let a = if axis < 0 {
            (axis + rank) as usize
        } else {
            axis as usize
        };
        let mut starts = vec![0i64; self.rank()];
        let mut limits: Vec<i64> = self.shape.dims.clone();
        starts[a] = start;
        limits[a] = start + len;
        self.slice_range(&starts, &limits)
    }

    // ── Comparison ops ──────────────────────────────────────────────

    fn compare_op(&self, other: &Tensor, dir: CompareDirection) -> Result<Tensor, Error> {
        self.check_same_graph(other)?;
        let mut b = self.graph.borrow_mut();
        let (shape, value) = b.compare(&self.shape, self.value, &other.shape, other.value, dir)?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }

    pub fn eq(&self, other: &Tensor) -> Result<Tensor, Error> {
        self.compare_op(other, CompareDirection::EQ)
    }

    pub fn ne(&self, other: &Tensor) -> Result<Tensor, Error> {
        self.compare_op(other, CompareDirection::NE)
    }

    pub fn lt(&self, other: &Tensor) -> Result<Tensor, Error> {
        self.compare_op(other, CompareDirection::LT)
    }

    pub fn le(&self, other: &Tensor) -> Result<Tensor, Error> {
        self.compare_op(other, CompareDirection::LE)
    }

    pub fn gt(&self, other: &Tensor) -> Result<Tensor, Error> {
        self.compare_op(other, CompareDirection::GT)
    }

    pub fn ge(&self, other: &Tensor) -> Result<Tensor, Error> {
        self.compare_op(other, CompareDirection::GE)
    }

    // ── Select ──────────────────────────────────────────────────────

    pub fn select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> Result<Tensor, Error> {
        if !Rc::ptr_eq(&pred.graph, &on_true.graph) || !Rc::ptr_eq(&pred.graph, &on_false.graph) {
            return Err(Error::GraphMismatch);
        }
        let mut b = pred.graph.borrow_mut();
        let (shape, value) = b.select(
            &pred.shape,
            pred.value,
            &on_true.shape,
            on_true.value,
            &on_false.shape,
            on_false.value,
        )?;
        drop(b);
        Ok(Tensor::new(shape, value, &pred.graph))
    }

    // ── Gather (embedding lookup) ───────────────────────────────────

    pub fn gather(&self, indices: &Tensor) -> Result<Tensor, Error> {
        self.check_same_graph(indices)?;
        let mut b = self.graph.borrow_mut();
        let (shape, value) =
            b.embedding_lookup(&self.shape, self.value, &indices.shape, indices.value)?;
        drop(b);
        Ok(Tensor::new(shape, value, &self.graph))
    }
}

// ── Trait impls ─────────────────────────────────────────────────────

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            shape: self.shape.clone(),
            value: self.value,
            graph: self.graph.clone(),
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("value", &self.value)
            .finish()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.value == other.value
    }
}

impl Eq for Tensor {}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} : {}", self.value, self.shape)
    }
}

// ── Operator overloads ──────────────────────────────────────────────
//
// All return Result<Tensor, Error> to propagate shape/broadcast errors.
// Use: `let c = (&a + &b)?;` or `let c = (a + b)?;`

macro_rules! impl_binary_op {
    ($trait:ident, $trait_method:ident, $tensor_method:ident) => {
        impl std::ops::$trait<&Tensor> for &Tensor {
            type Output = Result<Tensor, Error>;
            fn $trait_method(self, rhs: &Tensor) -> Result<Tensor, Error> {
                Tensor::$tensor_method(self, rhs)
            }
        }
        impl std::ops::$trait<&Tensor> for Tensor {
            type Output = Result<Tensor, Error>;
            fn $trait_method(self, rhs: &Tensor) -> Result<Tensor, Error> {
                Tensor::$tensor_method(&self, rhs)
            }
        }
        impl std::ops::$trait<Tensor> for &Tensor {
            type Output = Result<Tensor, Error>;
            fn $trait_method(self, rhs: Tensor) -> Result<Tensor, Error> {
                Tensor::$tensor_method(self, &rhs)
            }
        }
        impl std::ops::$trait<Tensor> for Tensor {
            type Output = Result<Tensor, Error>;
            fn $trait_method(self, rhs: Tensor) -> Result<Tensor, Error> {
                Tensor::$tensor_method(&self, &rhs)
            }
        }
    };
}

impl_binary_op!(Add, add, add);
impl_binary_op!(Sub, sub, sub);
impl_binary_op!(Mul, mul, mul);
impl_binary_op!(Div, div, div);

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(&self)
    }
}

// ── Scalar operator overloads ───────────────────────────────────────
//
// Enable `tensor * 2.0`, `2.0 * tensor`, etc.

macro_rules! impl_scalar_op {
    ($trait:ident, $trait_method:ident, $tensor_method:ident) => {
        impl std::ops::$trait<f64> for &Tensor {
            type Output = Result<Tensor, Error>;
            fn $trait_method(self, rhs: f64) -> Result<Tensor, Error> {
                Tensor::$tensor_method(self, rhs)
            }
        }
        impl std::ops::$trait<f64> for Tensor {
            type Output = Result<Tensor, Error>;
            fn $trait_method(self, rhs: f64) -> Result<Tensor, Error> {
                Tensor::$tensor_method(&self, rhs)
            }
        }
    };
}

impl_scalar_op!(Add, add, add_scalar);
impl_scalar_op!(Sub, sub, sub_scalar);
impl_scalar_op!(Mul, mul, mul_scalar);
impl_scalar_op!(Div, div, div_scalar);

macro_rules! impl_scalar_op_reverse {
    ($trait:ident, $trait_method:ident, $tensor_method:ident) => {
        impl std::ops::$trait<&Tensor> for f64 {
            type Output = Result<Tensor, Error>;
            fn $trait_method(self, rhs: &Tensor) -> Result<Tensor, Error> {
                rhs.$tensor_method(self)
            }
        }
        impl std::ops::$trait<Tensor> for f64 {
            type Output = Result<Tensor, Error>;
            fn $trait_method(self, rhs: Tensor) -> Result<Tensor, Error> {
                rhs.$tensor_method(self)
            }
        }
    };
}

impl_scalar_op_reverse!(Mul, mul, mul_scalar);
impl_scalar_op_reverse!(Add, add, add_scalar);

// f64 - tensor = -(tensor - f64) ... but simpler: create scalar and subtract
impl std::ops::Sub<&Tensor> for f64 {
    type Output = Result<Tensor, Error>;
    fn sub(self, rhs: &Tensor) -> Result<Tensor, Error> {
        let s = rhs.scalar_const(self);
        s.sub(rhs)
    }
}

impl std::ops::Sub<Tensor> for f64 {
    type Output = Result<Tensor, Error>;
    fn sub(self, rhs: Tensor) -> Result<Tensor, Error> {
        let s = rhs.scalar_const(self);
        s.sub(&rhs)
    }
}

impl std::ops::Div<&Tensor> for f64 {
    type Output = Result<Tensor, Error>;
    fn div(self, rhs: &Tensor) -> Result<Tensor, Error> {
        let s = rhs.scalar_const(self);
        s.div(rhs)
    }
}

impl std::ops::Div<Tensor> for f64 {
    type Output = Result<Tensor, Error>;
    fn div(self, rhs: Tensor) -> Result<Tensor, Error> {
        let s = rhs.scalar_const(self);
        s.div(&rhs)
    }
}
