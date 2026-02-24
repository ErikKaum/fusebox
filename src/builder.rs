// defines FuncBuilder: creates params (which are Tensors), appends instructions, does shape inference/checks,
// and returns new Tensors.

use crate::error::Error;
use crate::ir::{
    Add, BroadcastInDim, DotGeneral, Function, Inst, Logistic, Multiply, Param, ParamKind, Stmt,
};
use crate::shape::Shape;
use crate::tensor::Tensor;
use crate::value::ValueId;

/// Builds a single function (graph) by appending SSA instructions.
pub struct FuncBuilder {
    func: Function,
    next_id: u32,
}

impl FuncBuilder {
    /// Start building a new function with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            func: Function::new(name),
            next_id: 0,
        }
    }

    /// Finish building and return the Function IR.
    pub fn finish(self) -> Function {
        self.func
    }

    fn fresh(&mut self) -> ValueId {
        let id = self.next_id;
        self.next_id += 1;
        ValueId(id)
    }

    pub fn input(&mut self, name: impl Into<String>, shape: Shape) -> Tensor {
        self.param(name, shape, ParamKind::Input)
    }

    pub fn weight(&mut self, name: impl Into<String>, shape: Shape) -> Tensor {
        self.param(name, shape, ParamKind::Weight)
    }

    fn param(&mut self, name: impl Into<String>, shape: Shape, kind: ParamKind) -> Tensor {
        let id = self.fresh();
        self.func.params.push(Param {
            name: name.into(),
            shape: shape.clone(),
            value: id,
            kind,
        });
        Tensor::new(shape, id)
    }

    /// Set the function return value.
    pub fn ret(&mut self, t: &Tensor) {
        self.func.ret = Some(t.value);
    }

    /// 2D matmul using stablehlo.dot_general.
    ///
    /// Scope: only supports:
    ///   x: [B, In], w: [In, Out]  =>  y: [B, Out]
    pub fn matmul_2d(&mut self, x: &Tensor, w: &Tensor) -> Result<Tensor, Error> {
        const OP: &str = "matmul_2d";

        if x.dtype() != w.dtype() {
            return Err(Error::dtype(OP, x.dtype(), w.dtype()));
        }
        if x.rank() != 2 {
            return Err(Error::rank(OP, 2, x.rank()));
        }
        if w.rank() != 2 {
            return Err(Error::rank(OP, 2, w.rank()));
        }

        let b = x.shape.dim(0);
        let in_x = x.shape.dim(1);
        let in_w = w.shape.dim(0);
        let out = w.shape.dim(1);

        if in_x != in_w {
            return Err(Error::DimMismatch {
                op: OP,
                axis_a: 1,
                dim_a: in_x,
                axis_b: 0,
                dim_b: in_w,
            });
        }

        let out_shape = Shape::new(vec![b, out], x.dtype());
        let result = self.fresh();

        self.func.insts.push(Stmt {
            result,
            inst: Inst::DotGeneral(DotGeneral {
                lhs: x.value,
                rhs: w.value,
                out: out_shape.clone(),
                contracting_dims_lhs: vec![1],
                contracting_dims_rhs: vec![0],
                batching_dims_lhs: vec![],
                batching_dims_rhs: vec![],
            }),
        });

        Ok(Tensor::new(out_shape, result))
    }

    /// Broadcast a 1D bias [Out] into [B, Out] for adding to a matmul result.
    ///
    /// Scope: only supports:
    ///   b: [Out]  =>  bb: [B, Out] with dims=[1]
    pub fn broadcast_bias_1d(&mut self, b: &Tensor, batch: i64) -> Result<Tensor, Error> {
        const OP: &str = "broadcast_bias_1d";

        if b.rank() != 1 {
            return Err(Error::rank(OP, 1, b.rank()));
        }
        let out = b.shape.dim(0);
        let out_shape = Shape::new(vec![batch, out], b.dtype());
        let result = self.fresh();

        self.func.insts.push(Stmt {
            result,
            inst: Inst::BroadcastInDim(BroadcastInDim {
                operand: b.value,
                out: out_shape.clone(),
                dims: vec![1],
            }),
        });

        Ok(Tensor::new(out_shape, result))
    }

    /// Elementwise add. For now, shapes must match exactly.
    pub fn add(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
        const OP: &str = "add";

        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                op: OP,
                a: a.dtype(),
                b: b.dtype(),
            });
        }
        if a.shape != b.shape {
            return Err(Error::shape(OP, &a.shape, &b.shape));
        }

        let out_shape = a.shape.clone();
        let result = self.fresh();

        self.func.insts.push(Stmt {
            result,
            inst: Inst::Add(Add {
                lhs: a.value,
                rhs: b.value,
                out: out_shape.clone(),
            }),
        });
        Ok(Tensor::new(out_shape, result))
    }

    /// Elementwise multiply. Shapes must match exactly.
    pub fn mul(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
        const OP: &str = "mul";

        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                op: OP,
                a: a.dtype(),
                b: b.dtype(),
            });
        }
        if a.shape != b.shape {
            return Err(Error::shape(OP, &a.shape, &b.shape));
        }

        let out_shape = a.shape.clone();
        let result = self.fresh();

        self.func.insts.push(Stmt {
            result,
            inst: Inst::Multiply(Multiply {
                lhs: a.value,
                rhs: b.value,
                out: out_shape.clone(),
            }),
        });
        Ok(Tensor::new(out_shape, result))
    }

    /// SiLU activation: x * sigmoid(x).
    /// Emits a logistic (sigmoid) followed by an elementwise multiply.
    pub fn silu(&mut self, x: &Tensor) -> Result<Tensor, Error> {
        let sig_shape = x.shape.clone();
        let sig_result = self.fresh();
        self.func.insts.push(Stmt {
            result: sig_result,
            inst: Inst::Logistic(Logistic {
                operand: x.value,
                out: sig_shape.clone(),
            }),
        });
        let sig = Tensor::new(sig_shape, sig_result);
        self.mul(x, &sig)
    }
}
