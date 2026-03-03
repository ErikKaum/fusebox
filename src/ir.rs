//! Intermediate representation for computation graphs.
//!
//! The IR mirrors StableHLO: each [`Inst`] variant maps 1:1 to a StableHLO op.
//! A [`Function`] is a flat list of SSA statements with typed parameters and
//! explicit return values. [`Module`] groups one or more functions for MLIR printing.

use serde::{Deserialize, Serialize};

use crate::{shape::Shape, value::ValueId};

/// A collection of functions forming a complete MLIR module.
#[derive(Debug, Clone, Default)]
pub struct Module {
    pub functions: Vec<Function>,
}

/// A single function in the IR: named parameters, a body of SSA statements, and return values.
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<Param>,
    pub insts: Vec<Stmt>,
    pub returns: Vec<ValueId>,
}

impl Function {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            insts: Vec::new(),
            returns: Vec::new(),
        }
    }
}

/// Distinguishes runtime inputs from pre-loaded weights.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParamKind {
    Input,
    Weight,
}

/// A named, typed function parameter (either an input or a weight).
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub shape: Shape,
    pub value: ValueId,
    pub kind: ParamKind,
}

/// One SSA statement: produces `result` via `inst`.
#[derive(Debug, Clone)]
pub struct Stmt {
    pub result: ValueId,
    pub inst: Inst,
}

// ── Op payload structs ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BinaryOp {
    pub lhs: ValueId,
    pub rhs: ValueId,
    pub out: Shape,
}

#[derive(Debug, Clone)]
pub struct UnaryOp {
    pub operand: ValueId,
    pub out: Shape,
}

/// Batched matrix multiply (StableHLO `dot_general`).
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

/// Broadcasts a tensor into a larger shape along the given dimension mapping.
#[derive(Debug, Clone)]
pub struct BroadcastInDim {
    pub operand: ValueId,
    pub out: Shape,
    /// Maps each source dimension to its position in the output shape.
    pub dims: Vec<i64>,
}

/// A splat constant: a single scalar value broadcast to `out` shape.
#[derive(Debug, Clone)]
pub struct Constant {
    pub value: f64,
    pub out: Shape,
}

#[derive(Debug, Clone)]
pub struct TransposeOp {
    pub operand: ValueId,
    pub permutation: Vec<i64>,
    pub out: Shape,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    Sum,
    Max,
    Min,
}

#[derive(Debug, Clone)]
pub struct Reduce {
    pub operand: ValueId,
    pub init_value: ValueId,
    pub dimensions: Vec<i64>,
    pub kind: ReduceKind,
    pub out: Shape,
}

#[derive(Debug, Clone)]
pub struct Concatenate {
    pub operands: Vec<ValueId>,
    pub dimension: i64,
    pub out: Shape,
}

#[derive(Debug, Clone)]
pub struct Slice {
    pub operand: ValueId,
    pub start_indices: Vec<i64>,
    pub limit_indices: Vec<i64>,
    pub strides: Vec<i64>,
    pub out: Shape,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CompareDirection {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
}

impl CompareDirection {
    pub fn mlir_str(self) -> &'static str {
        match self {
            CompareDirection::EQ => "EQ",
            CompareDirection::NE => "NE",
            CompareDirection::LT => "LT",
            CompareDirection::LE => "LE",
            CompareDirection::GT => "GT",
            CompareDirection::GE => "GE",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompareOp {
    pub lhs: ValueId,
    pub rhs: ValueId,
    pub direction: CompareDirection,
    pub out: Shape,
}

#[derive(Debug, Clone)]
pub struct SelectOp {
    pub pred: ValueId,
    pub on_true: ValueId,
    pub on_false: ValueId,
    pub out: Shape,
}

#[derive(Debug, Clone)]
pub struct IotaOp {
    pub iota_dimension: i64,
    pub out: Shape,
}

/// Argmax via a paired value+index reduce (StableHLO pattern).
#[derive(Debug, Clone)]
pub struct ReduceArgMax {
    pub operand: ValueId,
    pub iota: ValueId,
    pub init_value: ValueId,
    pub init_index: ValueId,
    pub dimension: i64,
    pub out: Shape,
}

/// Embedding lookup / advanced indexing (StableHLO `gather`).
#[derive(Debug, Clone)]
pub struct GatherOp {
    pub operand: ValueId,
    pub start_indices: ValueId,
    pub offset_dims: Vec<i64>,
    pub collapsed_slice_dims: Vec<i64>,
    pub start_index_map: Vec<i64>,
    pub index_vector_dim: i64,
    pub slice_sizes: Vec<i64>,
    pub out: Shape,
}

/// Every operation the graph can contain. Each variant maps to one StableHLO op.
#[derive(Debug, Clone)]
pub enum Inst {
    DotGeneral(DotGeneral),
    BroadcastInDim(BroadcastInDim),

    // Binary arithmetic
    Add(BinaryOp),
    Subtract(BinaryOp),
    Multiply(BinaryOp),
    Divide(BinaryOp),
    Maximum(BinaryOp),

    // Unary
    Negate(UnaryOp),
    Exponential(UnaryOp),
    Log(UnaryOp),
    Sqrt(UnaryOp),
    Rsqrt(UnaryOp),
    Abs(UnaryOp),
    Tanh(UnaryOp),
    Logistic(UnaryOp),
    Cosine(UnaryOp),
    Sine(UnaryOp),

    // Type conversion
    Convert(UnaryOp),

    Constant(Constant),

    // Shape manipulation
    Reshape(UnaryOp),
    Transpose(TransposeOp),
    Concatenate(Concatenate),
    Slice(Slice),

    // Reductions
    Reduce(Reduce),
    ReduceArgMax(ReduceArgMax),

    // Comparison and selection
    Compare(CompareOp),
    Select(SelectOp),

    // Index generation
    Iota(IotaOp),

    // Gather (embedding lookup)
    Gather(GatherOp),
}
