// src/ir.rs defines a tiny internal IR: a Function with params, a list of instructions (result_id, Inst),
// and a return value; and Inst variants for the three ops we support.

use crate::{shape::Shape, value::ValueId};

/// A whole module. Usually just a single function.
#[derive(Debug, Clone, Default)]
pub struct Module {
    pub functions: Vec<Function>,
}

/// A function in our tiny IR.
///
/// Params are SSA values too: each parameter gets a ValueId.
/// Instructions are SSA-producing: each instruction produces one result ValueId.
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<Param>,
    pub insts: Vec<Stmt>,
    pub ret: Option<ValueId>,
}

impl Function {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            insts: Vec::new(),
            ret: None,
        }
    }
}

/// A function parameter.
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub shape: Shape,
    pub value: ValueId,
}

/// One SSA statement: `%result = inst(...)`
#[derive(Debug, Clone)]
pub struct Stmt {
    pub result: ValueId,
    pub inst: Inst,
}

/// The only operations we support initially:
/// - a 2D matmul via stablehlo.dot_general
/// - broadcast_in_dim for bias
/// - elementwise add
#[derive(Debug, Clone)]
pub enum Inst {
    DotGeneral(DotGeneral),
    BroadcastInDim(BroadcastInDim),
    Add(Add),
}

/// DotGeneral config (enough to print stablehlo.dot_general attrs)
#[derive(Debug, Clone)]
pub struct DotGeneral {
    pub lhs: ValueId,
    pub rhs: ValueId,
    pub out: Shape,

    pub contracting_dims_lhs: Vec<i64>,
    pub contracting_dims_rhs: Vec<i64>,
    pub batching_dims_lhs: Vec<i64>,
    pub batching_dims_rhs: Vec<i64>,
}

/// BroadcastInDim config
#[derive(Debug, Clone)]
pub struct BroadcastInDim {
    pub operand: ValueId,
    pub out: Shape,
    pub dims: Vec<i64>,
}

/// Add config
#[derive(Debug, Clone)]
pub struct Add {
    pub lhs: ValueId,
    pub rhs: ValueId,
    pub out: Shape,
}
