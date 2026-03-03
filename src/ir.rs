use crate::{shape::Shape, value::ValueId};

#[derive(Debug, Clone, Default)]
pub struct Module {
    pub functions: Vec<Function>,
}

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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ParamKind {
    Input,
    Weight,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub shape: Shape,
    pub value: ValueId,
    pub kind: ParamKind,
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub result: ValueId,
    pub inst: Inst,
}

// ── Generalized op structs ──────────────────────────────────────────

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

#[derive(Debug, Clone)]
pub struct BroadcastInDim {
    pub operand: ValueId,
    pub out: Shape,
    pub dims: Vec<i64>,
}

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

#[derive(Debug, Clone)]
pub struct ReduceArgMax {
    pub operand: ValueId,
    pub iota: ValueId,
    pub init_value: ValueId,
    pub init_index: ValueId,
    pub dimension: i64,
    pub out: Shape,
}

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
