use crate::dtype::DType;
use crate::error::Error;
use crate::ir::{
    BinaryOp, BroadcastInDim, CompareDirection, CompareOp, Concatenate, Constant, DotGeneral,
    Function, GatherOp, Inst, IotaOp, Param, ParamKind, Reduce, ReduceArgMax, ReduceKind, SelectOp,
    Slice, Stmt, TransposeOp, UnaryOp,
};
use crate::shape::Shape;
use crate::value::ValueId;

pub struct FuncBuilder {
    func: Function,
    next_id: u32,
    scope: Vec<String>,
}

impl FuncBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            func: Function::new(name),
            next_id: 0,
            scope: Vec::new(),
        }
    }

    pub fn push_scope(&mut self, name: &str) {
        self.scope.push(name.to_string());
    }

    pub fn pop_scope(&mut self) {
        self.scope.pop();
    }

    pub fn current_scope(&self) -> String {
        self.scope.join("/")
    }

    pub fn into_function(self) -> Function {
        self.func
    }

    pub fn function(&self) -> &Function {
        &self.func
    }

    pub fn fresh(&mut self) -> ValueId {
        let id = self.next_id;
        self.next_id += 1;
        ValueId(id)
    }

    // ── Parameter creation ──────────────────────────────────────────

    pub fn add_input(&mut self, name: String, shape: Shape) -> ValueId {
        self.add_param(name, shape, ParamKind::Input)
    }

    pub fn add_weight(&mut self, name: String, shape: Shape) -> ValueId {
        self.add_param(name, shape, ParamKind::Weight)
    }

    fn add_param(&mut self, name: String, shape: Shape, kind: ParamKind) -> ValueId {
        let id = self.fresh();
        self.func.params.push(Param {
            name,
            shape,
            value: id,
            kind,
        });
        id
    }

    pub fn set_return(&mut self, val: ValueId) {
        self.func.returns = vec![val];
    }

    pub fn set_returns(&mut self, vals: Vec<ValueId>) {
        self.func.returns = vals;
    }

    // ── Broadcasting ────────────────────────────────────────────────

    fn broadcast_pair(
        &mut self,
        a_shape: &Shape,
        a_val: ValueId,
        b_shape: &Shape,
        b_val: ValueId,
    ) -> Result<(Shape, ValueId, ValueId), Error> {
        if a_shape.dtype != b_shape.dtype {
            return Err(Error::dtype("broadcast", a_shape.dtype, b_shape.dtype));
        }

        let (out_shape, a_dims, b_dims) = compute_broadcast(a_shape, b_shape)?;

        let a_val = if a_shape.dims == out_shape.dims {
            a_val
        } else {
            let result = self.fresh();
            self.func.insts.push(Stmt {
                result,
                inst: Inst::BroadcastInDim(BroadcastInDim {
                    operand: a_val,
                    out: out_shape.clone(),
                    dims: a_dims,
                }),
            });
            result
        };

        let b_val = if b_shape.dims == out_shape.dims {
            b_val
        } else {
            let result = self.fresh();
            self.func.insts.push(Stmt {
                result,
                inst: Inst::BroadcastInDim(BroadcastInDim {
                    operand: b_val,
                    out: out_shape.clone(),
                    dims: b_dims,
                }),
            });
            result
        };

        Ok((out_shape, a_val, b_val))
    }

    // ── Binary ops (with auto-broadcast) ────────────────────────────

    fn binary_op(
        &mut self,
        a_shape: &Shape,
        a_val: ValueId,
        b_shape: &Shape,
        b_val: ValueId,
        make_inst: fn(BinaryOp) -> Inst,
    ) -> Result<(Shape, ValueId), Error> {
        let (out_shape, a_val, b_val) = self.broadcast_pair(a_shape, a_val, b_shape, b_val)?;
        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: make_inst(BinaryOp {
                lhs: a_val,
                rhs: b_val,
                out: out_shape.clone(),
            }),
        });
        Ok((out_shape, result))
    }

    pub fn add(
        &mut self,
        a_shape: &Shape,
        a_val: ValueId,
        b_shape: &Shape,
        b_val: ValueId,
    ) -> Result<(Shape, ValueId), Error> {
        self.binary_op(a_shape, a_val, b_shape, b_val, Inst::Add)
    }

    pub fn sub(
        &mut self,
        a_shape: &Shape,
        a_val: ValueId,
        b_shape: &Shape,
        b_val: ValueId,
    ) -> Result<(Shape, ValueId), Error> {
        self.binary_op(a_shape, a_val, b_shape, b_val, Inst::Subtract)
    }

    pub fn mul(
        &mut self,
        a_shape: &Shape,
        a_val: ValueId,
        b_shape: &Shape,
        b_val: ValueId,
    ) -> Result<(Shape, ValueId), Error> {
        self.binary_op(a_shape, a_val, b_shape, b_val, Inst::Multiply)
    }

    pub fn div(
        &mut self,
        a_shape: &Shape,
        a_val: ValueId,
        b_shape: &Shape,
        b_val: ValueId,
    ) -> Result<(Shape, ValueId), Error> {
        self.binary_op(a_shape, a_val, b_shape, b_val, Inst::Divide)
    }

    pub fn maximum(
        &mut self,
        a_shape: &Shape,
        a_val: ValueId,
        b_shape: &Shape,
        b_val: ValueId,
    ) -> Result<(Shape, ValueId), Error> {
        self.binary_op(a_shape, a_val, b_shape, b_val, Inst::Maximum)
    }

    // ── Unary ops ───────────────────────────────────────────────────

    fn unary_op(
        &mut self,
        shape: &Shape,
        val: ValueId,
        make_inst: fn(UnaryOp) -> Inst,
    ) -> (Shape, ValueId) {
        let out = shape.clone();
        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: make_inst(UnaryOp {
                operand: val,
                out: out.clone(),
            }),
        });
        (out, result)
    }

    pub fn neg(&mut self, shape: &Shape, val: ValueId) -> (Shape, ValueId) {
        self.unary_op(shape, val, Inst::Negate)
    }

    pub fn exp(&mut self, shape: &Shape, val: ValueId) -> (Shape, ValueId) {
        self.unary_op(shape, val, Inst::Exponential)
    }

    pub fn log(&mut self, shape: &Shape, val: ValueId) -> (Shape, ValueId) {
        self.unary_op(shape, val, Inst::Log)
    }

    pub fn sqrt(&mut self, shape: &Shape, val: ValueId) -> (Shape, ValueId) {
        self.unary_op(shape, val, Inst::Sqrt)
    }

    pub fn rsqrt(&mut self, shape: &Shape, val: ValueId) -> (Shape, ValueId) {
        self.unary_op(shape, val, Inst::Rsqrt)
    }

    pub fn abs(&mut self, shape: &Shape, val: ValueId) -> (Shape, ValueId) {
        self.unary_op(shape, val, Inst::Abs)
    }

    pub fn tanh(&mut self, shape: &Shape, val: ValueId) -> (Shape, ValueId) {
        self.unary_op(shape, val, Inst::Tanh)
    }

    pub fn logistic(&mut self, shape: &Shape, val: ValueId) -> (Shape, ValueId) {
        self.unary_op(shape, val, Inst::Logistic)
    }

    pub fn cosine(&mut self, shape: &Shape, val: ValueId) -> (Shape, ValueId) {
        self.unary_op(shape, val, Inst::Cosine)
    }

    pub fn sine(&mut self, shape: &Shape, val: ValueId) -> (Shape, ValueId) {
        self.unary_op(shape, val, Inst::Sine)
    }

    pub fn convert(
        &mut self,
        shape: &Shape,
        val: ValueId,
        target_dtype: DType,
    ) -> (Shape, ValueId) {
        let out = Shape::new(shape.dims.clone(), target_dtype);
        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::Convert(UnaryOp {
                operand: val,
                out: out.clone(),
            }),
        });
        (out, result)
    }

    // ── Composite activations ───────────────────────────────────────

    pub fn silu(&mut self, shape: &Shape, val: ValueId) -> (Shape, ValueId) {
        let (sig_shape, sig_val) = self.logistic(shape, val);
        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::Multiply(BinaryOp {
                lhs: val,
                rhs: sig_val,
                out: sig_shape.clone(),
            }),
        });
        (sig_shape, result)
    }

    pub fn relu(&mut self, shape: &Shape, val: ValueId) -> (Shape, ValueId) {
        let zero_val = self.emit_constant(0.0, shape);
        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::Maximum(BinaryOp {
                lhs: val,
                rhs: zero_val,
                out: shape.clone(),
            }),
        });
        (shape.clone(), result)
    }

    /// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(&mut self, shape: &Shape, val: ValueId) -> (Shape, ValueId) {
        let half = self.emit_constant(0.5, shape);
        let coeff = self.emit_constant(0.044715, shape);
        let sqrt_2_over_pi = self.emit_constant(0.7978845608028654, shape);
        let one = self.emit_constant(1.0, shape);

        // x^3 = x * x * x
        let x_sq = self.fresh();
        self.func.insts.push(Stmt {
            result: x_sq,
            inst: Inst::Multiply(BinaryOp {
                lhs: val,
                rhs: val,
                out: shape.clone(),
            }),
        });
        let x_cu = self.fresh();
        self.func.insts.push(Stmt {
            result: x_cu,
            inst: Inst::Multiply(BinaryOp {
                lhs: x_sq,
                rhs: val,
                out: shape.clone(),
            }),
        });

        // coeff * x^3
        let cx3 = self.fresh();
        self.func.insts.push(Stmt {
            result: cx3,
            inst: Inst::Multiply(BinaryOp {
                lhs: coeff,
                rhs: x_cu,
                out: shape.clone(),
            }),
        });

        // x + coeff * x^3
        let inner_sum = self.fresh();
        self.func.insts.push(Stmt {
            result: inner_sum,
            inst: Inst::Add(BinaryOp {
                lhs: val,
                rhs: cx3,
                out: shape.clone(),
            }),
        });

        // sqrt(2/pi) * (x + coeff * x^3)
        let scaled = self.fresh();
        self.func.insts.push(Stmt {
            result: scaled,
            inst: Inst::Multiply(BinaryOp {
                lhs: sqrt_2_over_pi,
                rhs: inner_sum,
                out: shape.clone(),
            }),
        });

        // tanh(...)
        let (_, tanh_val) = self.tanh(shape, scaled);

        // 1 + tanh(...)
        let one_plus = self.fresh();
        self.func.insts.push(Stmt {
            result: one_plus,
            inst: Inst::Add(BinaryOp {
                lhs: one,
                rhs: tanh_val,
                out: shape.clone(),
            }),
        });

        // 0.5 * x
        let half_x = self.fresh();
        self.func.insts.push(Stmt {
            result: half_x,
            inst: Inst::Multiply(BinaryOp {
                lhs: half,
                rhs: val,
                out: shape.clone(),
            }),
        });

        // 0.5 * x * (1 + tanh(...))
        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::Multiply(BinaryOp {
                lhs: half_x,
                rhs: one_plus,
                out: shape.clone(),
            }),
        });

        (shape.clone(), result)
    }

    // ── Constants ───────────────────────────────────────────────────

    pub fn constant(&mut self, value: f64, shape: &Shape) -> (Shape, ValueId) {
        let val = self.emit_constant(value, shape);
        (shape.clone(), val)
    }

    fn emit_constant(&mut self, value: f64, shape: &Shape) -> ValueId {
        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::Constant(Constant {
                value,
                out: shape.clone(),
            }),
        });
        result
    }

    // ── Shape manipulation ──────────────────────────────────────────

    pub fn reshape(
        &mut self,
        shape: &Shape,
        val: ValueId,
        new_dims: &[i64],
    ) -> Result<(Shape, ValueId), Error> {
        let old_numel: i64 = shape.dims.iter().product();
        let new_numel: i64 = new_dims.iter().product();
        if old_numel != new_numel {
            return Err(Error::InvalidParam {
                msg: format!("reshape: element count mismatch (old={old_numel}, new={new_numel})"),
            });
        }
        let out_shape = Shape::new(new_dims.to_vec(), shape.dtype);
        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::Reshape(UnaryOp {
                operand: val,
                out: out_shape.clone(),
            }),
        });
        Ok((out_shape, result))
    }

    pub fn transpose(
        &mut self,
        shape: &Shape,
        val: ValueId,
        permutation: &[i64],
    ) -> Result<(Shape, ValueId), Error> {
        if permutation.len() != shape.rank() {
            return Err(Error::InvalidParam {
                msg: format!(
                    "transpose: permutation length {} != rank {}",
                    permutation.len(),
                    shape.rank()
                ),
            });
        }
        let out_dims: Vec<i64> = permutation
            .iter()
            .map(|&p| shape.dims[p as usize])
            .collect();
        let out_shape = Shape::new(out_dims, shape.dtype);
        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::Transpose(TransposeOp {
                operand: val,
                permutation: permutation.to_vec(),
                out: out_shape.clone(),
            }),
        });
        Ok((out_shape, result))
    }

    // ── Expand (broadcast size-1 dims) ────────────────────────────

    pub fn expand(
        &mut self,
        shape: &Shape,
        val: ValueId,
        target_dims: &[i64],
    ) -> Result<(Shape, ValueId), Error> {
        if target_dims.len() != shape.rank() {
            return Err(Error::InvalidParam {
                msg: format!(
                    "expand: target rank {} != source rank {}",
                    target_dims.len(),
                    shape.rank()
                ),
            });
        }
        for (i, (&src, &tgt)) in shape.dims.iter().zip(target_dims.iter()).enumerate() {
            if src != tgt && src != 1 {
                return Err(Error::InvalidParam {
                    msg: format!(
                        "expand: dim {} has size {} (not 1), cannot expand to {}",
                        i, src, tgt
                    ),
                });
            }
        }
        if shape.dims == target_dims {
            return Ok((shape.clone(), val));
        }
        let out_shape = Shape::new(target_dims.to_vec(), shape.dtype);
        let dims: Vec<i64> = (0..shape.rank() as i64).collect();
        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::BroadcastInDim(BroadcastInDim {
                operand: val,
                out: out_shape.clone(),
                dims,
            }),
        });
        Ok((out_shape, result))
    }

    // ── Reductions ──────────────────────────────────────────────────

    fn reduce(
        &mut self,
        shape: &Shape,
        val: ValueId,
        axes: &[i64],
        kind: ReduceKind,
    ) -> Result<(Shape, ValueId), Error> {
        let rank = shape.rank();
        let mut normalized_axes = Vec::with_capacity(axes.len());
        for &axis in axes {
            let a = normalize_axis(axis, rank)?;
            normalized_axes.push(a as i64);
        }

        let mut out_dims = Vec::new();
        for (i, &d) in shape.dims.iter().enumerate() {
            if !normalized_axes.contains(&(i as i64)) {
                out_dims.push(d);
            }
        }
        let out_shape = Shape::new(out_dims, shape.dtype);

        let init_value = match kind {
            ReduceKind::Sum => 0.0_f64,
            ReduceKind::Max => f64::NEG_INFINITY,
            ReduceKind::Min => f64::INFINITY,
        };
        let init_shape = Shape::new(vec![], shape.dtype);
        let init_id = self.emit_constant(init_value, &init_shape);

        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::Reduce(Reduce {
                operand: val,
                init_value: init_id,
                dimensions: normalized_axes,
                kind,
                out: out_shape.clone(),
            }),
        });

        Ok((out_shape, result))
    }

    pub fn reduce_sum(
        &mut self,
        shape: &Shape,
        val: ValueId,
        axes: &[i64],
    ) -> Result<(Shape, ValueId), Error> {
        self.reduce(shape, val, axes, ReduceKind::Sum)
    }

    pub fn reduce_max(
        &mut self,
        shape: &Shape,
        val: ValueId,
        axes: &[i64],
    ) -> Result<(Shape, ValueId), Error> {
        self.reduce(shape, val, axes, ReduceKind::Max)
    }

    pub fn reduce_min(
        &mut self,
        shape: &Shape,
        val: ValueId,
        axes: &[i64],
    ) -> Result<(Shape, ValueId), Error> {
        self.reduce(shape, val, axes, ReduceKind::Min)
    }

    pub fn argmax(
        &mut self,
        shape: &Shape,
        val: ValueId,
        axis: i64,
    ) -> Result<(Shape, ValueId), Error> {
        let rank = shape.rank();
        let a = normalize_axis(axis, rank)?;

        let mut out_dims = Vec::new();
        for (i, &d) in shape.dims.iter().enumerate() {
            if i != a {
                out_dims.push(d);
            }
        }
        let out_shape = Shape::new(out_dims, DType::I32);

        let iota_shape = Shape::new(shape.dims.clone(), DType::I32);
        let iota_id = self.fresh();
        self.func.insts.push(Stmt {
            result: iota_id,
            inst: Inst::Iota(IotaOp {
                iota_dimension: a as i64,
                out: iota_shape,
            }),
        });

        let init_val_shape = Shape::new(vec![], shape.dtype);
        let init_val_id = self.emit_constant(f64::NEG_INFINITY, &init_val_shape);

        let init_idx_shape = Shape::new(vec![], DType::I32);
        let init_idx_id = self.fresh();
        self.func.insts.push(Stmt {
            result: init_idx_id,
            inst: Inst::Constant(Constant {
                value: 0.0,
                out: init_idx_shape,
            }),
        });

        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::ReduceArgMax(ReduceArgMax {
                operand: val,
                iota: iota_id,
                init_value: init_val_id,
                init_index: init_idx_id,
                dimension: a as i64,
                out: out_shape.clone(),
            }),
        });

        Ok((out_shape, result))
    }

    // ── Matmul (N-dimensional) ──────────────────────────────────────

    pub fn matmul(
        &mut self,
        x_shape: &Shape,
        x_val: ValueId,
        w_shape: &Shape,
        w_val: ValueId,
    ) -> Result<(Shape, ValueId), Error> {
        const OP: &str = "matmul";

        if x_shape.dtype != w_shape.dtype {
            return Err(Error::dtype(OP, x_shape.dtype, w_shape.dtype));
        }
        if x_shape.rank() < 2 {
            return Err(Error::rank(OP, 2, x_shape.rank()));
        }
        if w_shape.rank() < 2 {
            return Err(Error::rank(OP, 2, w_shape.rank()));
        }

        let x_rank = x_shape.rank();
        let w_rank = w_shape.rank();

        let m = x_shape.dims[x_rank - 2];
        let k_x = x_shape.dims[x_rank - 1];
        let k_w = w_shape.dims[w_rank - 2];
        let n = w_shape.dims[w_rank - 1];

        if k_x != k_w {
            return Err(Error::DimMismatch {
                op: OP,
                axis_a: x_rank - 1,
                dim_a: k_x,
                axis_b: w_rank - 2,
                dim_b: k_w,
            });
        }

        let x_batch = &x_shape.dims[..x_rank - 2];
        let w_batch = &w_shape.dims[..w_rank - 2];

        let (batch_out, actual_x_val, actual_w_val, eff_rank) = if x_batch == w_batch {
            (x_batch.to_vec(), x_val, w_val, x_rank)
        } else if w_batch.is_empty() {
            let mut new_w_dims = x_batch.to_vec();
            new_w_dims.push(k_w);
            new_w_dims.push(n);
            let new_w_shape = Shape::new(new_w_dims, w_shape.dtype);
            let bcast_dims: Vec<i64> = vec![x_batch.len() as i64, (x_batch.len() + 1) as i64];
            let bcast_id = self.fresh();
            self.func.insts.push(Stmt {
                result: bcast_id,
                inst: Inst::BroadcastInDim(BroadcastInDim {
                    operand: w_val,
                    out: new_w_shape,
                    dims: bcast_dims,
                }),
            });
            (x_batch.to_vec(), x_val, bcast_id, x_rank)
        } else if x_batch.is_empty() {
            let mut new_x_dims = w_batch.to_vec();
            new_x_dims.push(m);
            new_x_dims.push(k_x);
            let new_x_shape = Shape::new(new_x_dims, x_shape.dtype);
            let bcast_dims: Vec<i64> = vec![w_batch.len() as i64, (w_batch.len() + 1) as i64];
            let bcast_id = self.fresh();
            self.func.insts.push(Stmt {
                result: bcast_id,
                inst: Inst::BroadcastInDim(BroadcastInDim {
                    operand: x_val,
                    out: new_x_shape,
                    dims: bcast_dims,
                }),
            });
            (w_batch.to_vec(), bcast_id, w_val, w_rank)
        } else {
            return Err(Error::Unsupported {
                op: OP,
                msg: "incompatible batch dimensions for matmul",
            });
        };

        let mut out_dims = batch_out;
        out_dims.push(m);
        out_dims.push(n);
        let out_shape = Shape::new(out_dims, x_shape.dtype);

        let batching: Vec<i64> = (0..eff_rank as i64 - 2).collect();
        let contracting_lhs = vec![(eff_rank - 1) as i64];
        let contracting_rhs = vec![(eff_rank - 2) as i64];

        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::DotGeneral(DotGeneral {
                lhs: actual_x_val,
                rhs: actual_w_val,
                out: out_shape.clone(),
                contracting_dims_lhs: contracting_lhs,
                contracting_dims_rhs: contracting_rhs,
                batching_dims_lhs: batching.clone(),
                batching_dims_rhs: batching,
            }),
        });

        Ok((out_shape, result))
    }

    // ── Concatenate ─────────────────────────────────────────────────

    pub fn concatenate(
        &mut self,
        shapes: &[&Shape],
        vals: &[ValueId],
        axis: i64,
    ) -> Result<(Shape, ValueId), Error> {
        if shapes.is_empty() {
            return Err(Error::InvalidParam {
                msg: "concatenate: need at least one tensor".into(),
            });
        }
        let rank = shapes[0].rank();
        let a = normalize_axis(axis, rank)?;
        let dtype = shapes[0].dtype;

        for (i, s) in shapes.iter().enumerate().skip(1) {
            if s.dtype != dtype {
                return Err(Error::dtype("concatenate", dtype, s.dtype));
            }
            if s.rank() != rank {
                return Err(Error::InvalidParam {
                    msg: format!(
                        "concatenate: tensor {} has rank {}, expected {}",
                        i,
                        s.rank(),
                        rank
                    ),
                });
            }
            for d in 0..rank {
                if d != a && s.dims[d] != shapes[0].dims[d] {
                    return Err(Error::InvalidParam {
                        msg: format!(
                            "concatenate: dim {} mismatch between tensor 0 ({}) and tensor {} ({})",
                            d, shapes[0].dims[d], i, s.dims[d]
                        ),
                    });
                }
            }
        }

        let concat_size: i64 = shapes.iter().map(|s| s.dims[a]).sum();
        let mut out_dims = shapes[0].dims.clone();
        out_dims[a] = concat_size;
        let out_shape = Shape::new(out_dims, dtype);

        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::Concatenate(Concatenate {
                operands: vals.to_vec(),
                dimension: a as i64,
                out: out_shape.clone(),
            }),
        });
        Ok((out_shape, result))
    }

    // ── Slice ───────────────────────────────────────────────────────

    pub fn slice(
        &mut self,
        shape: &Shape,
        val: ValueId,
        start_indices: &[i64],
        limit_indices: &[i64],
        strides: &[i64],
    ) -> Result<(Shape, ValueId), Error> {
        let rank = shape.rank();
        if start_indices.len() != rank || limit_indices.len() != rank || strides.len() != rank {
            return Err(Error::InvalidParam {
                msg: format!("slice: indices length must match rank {}", rank),
            });
        }

        let mut out_dims = Vec::with_capacity(rank);
        for i in 0..rank {
            if start_indices[i] < 0 || limit_indices[i] > shape.dims[i] || strides[i] <= 0 {
                return Err(Error::InvalidParam {
                    msg: format!(
                        "slice: invalid range for dim {}: start={}, limit={}, stride={}, size={}",
                        i, start_indices[i], limit_indices[i], strides[i], shape.dims[i]
                    ),
                });
            }
            let size = (limit_indices[i] - start_indices[i] + strides[i] - 1) / strides[i];
            out_dims.push(size);
        }

        let out_shape = Shape::new(out_dims, shape.dtype);
        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::Slice(Slice {
                operand: val,
                start_indices: start_indices.to_vec(),
                limit_indices: limit_indices.to_vec(),
                strides: strides.to_vec(),
                out: out_shape.clone(),
            }),
        });
        Ok((out_shape, result))
    }

    // ── Comparison ──────────────────────────────────────────────────

    pub fn compare(
        &mut self,
        a_shape: &Shape,
        a_val: ValueId,
        b_shape: &Shape,
        b_val: ValueId,
        direction: CompareDirection,
    ) -> Result<(Shape, ValueId), Error> {
        let (broadcast_shape, a_val, b_val) =
            self.broadcast_pair(a_shape, a_val, b_shape, b_val)?;
        let out_shape = Shape::new(broadcast_shape.dims.clone(), DType::Bool);
        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::Compare(CompareOp {
                lhs: a_val,
                rhs: b_val,
                direction,
                out: out_shape.clone(),
            }),
        });
        Ok((out_shape, result))
    }

    // ── Select ──────────────────────────────────────────────────────

    pub fn select(
        &mut self,
        pred_shape: &Shape,
        pred_val: ValueId,
        true_shape: &Shape,
        true_val: ValueId,
        false_shape: &Shape,
        false_val: ValueId,
    ) -> Result<(Shape, ValueId), Error> {
        if pred_shape.dtype != DType::Bool {
            return Err(Error::InvalidParam {
                msg: format!("select: predicate must be Bool, got {}", pred_shape.dtype),
            });
        }
        if true_shape != false_shape {
            return Err(Error::ShapeMismatch {
                op: "select",
                a: true_shape.clone(),
                b: false_shape.clone(),
            });
        }
        let pred_dims_no_dtype = &pred_shape.dims;
        let true_dims = &true_shape.dims;
        if pred_dims_no_dtype != true_dims {
            return Err(Error::InvalidParam {
                msg: format!(
                    "select: pred dims {:?} != value dims {:?}",
                    pred_dims_no_dtype, true_dims
                ),
            });
        }

        let out_shape = true_shape.clone();
        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::Select(SelectOp {
                pred: pred_val,
                on_true: true_val,
                on_false: false_val,
                out: out_shape.clone(),
            }),
        });
        Ok((out_shape, result))
    }

    // ── Iota ────────────────────────────────────────────────────────

    pub fn iota(&mut self, shape: &Shape, dimension: i64) -> Result<(Shape, ValueId), Error> {
        let a = normalize_axis(dimension, shape.rank())?;
        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::Iota(IotaOp {
                iota_dimension: a as i64,
                out: shape.clone(),
            }),
        });
        Ok((shape.clone(), result))
    }

    // ── Gather (embedding lookup) ───────────────────────────────────

    pub fn embedding_lookup(
        &mut self,
        table_shape: &Shape,
        table_val: ValueId,
        indices_shape: &Shape,
        indices_val: ValueId,
    ) -> Result<(Shape, ValueId), Error> {
        if table_shape.rank() != 2 {
            return Err(Error::InvalidParam {
                msg: format!(
                    "embedding_lookup: table must be rank-2 [vocab, dim], got rank {}",
                    table_shape.rank()
                ),
            });
        }
        if !indices_shape.dtype.is_integer() {
            return Err(Error::InvalidParam {
                msg: format!(
                    "embedding_lookup: indices must be integer type, got {}",
                    indices_shape.dtype
                ),
            });
        }

        let embed_dim = table_shape.dims[1];
        let mut out_dims = indices_shape.dims.clone();
        out_dims.push(embed_dim);
        let out_shape = Shape::new(out_dims, table_shape.dtype);

        let index_vector_dim = indices_shape.rank() as i64;
        let slice_sizes = vec![1, embed_dim];

        let result = self.fresh();
        self.func.insts.push(Stmt {
            result,
            inst: Inst::Gather(GatherOp {
                operand: table_val,
                start_indices: indices_val,
                offset_dims: vec![index_vector_dim],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim,
                slice_sizes,
                out: out_shape.clone(),
            }),
        });
        Ok((out_shape, result))
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn compute_broadcast(a: &Shape, b: &Shape) -> Result<(Shape, Vec<i64>, Vec<i64>), Error> {
    let a_dims = &a.dims;
    let b_dims = &b.dims;
    let out_rank = a_dims.len().max(b_dims.len());

    let a_offset = out_rank - a_dims.len();
    let b_offset = out_rank - b_dims.len();

    let mut out_dims = Vec::with_capacity(out_rank);
    for i in 0..out_rank {
        let ad = if i >= a_offset {
            a_dims[i - a_offset]
        } else {
            1
        };
        let bd = if i >= b_offset {
            b_dims[i - b_offset]
        } else {
            1
        };

        if ad != bd && ad != 1 && bd != 1 {
            return Err(Error::BroadcastError {
                a: a.clone(),
                b: b.clone(),
            });
        }
        out_dims.push(ad.max(bd));
    }

    let a_map: Vec<i64> = (0..a_dims.len()).map(|i| (i + a_offset) as i64).collect();
    let b_map: Vec<i64> = (0..b_dims.len()).map(|i| (i + b_offset) as i64).collect();

    Ok((Shape::new(out_dims, a.dtype), a_map, b_map))
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize, Error> {
    let rank_i = rank as i64;
    let normalized = if axis < 0 { axis + rank_i } else { axis };
    if normalized < 0 || normalized >= rank_i {
        return Err(Error::InvalidParam {
            msg: format!("axis {axis} out of range for rank {rank}"),
        });
    }
    Ok(normalized as usize)
}
