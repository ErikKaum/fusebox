use core::fmt::Write;

use crate::dtype::DType;
use crate::ir::{Function, Inst, Module, ReduceKind};
use crate::shape::Shape;
use crate::value::ValueId;

pub fn print_module(module: &Module) -> String {
    let mut out = String::new();
    writeln!(&mut out, "module {{").unwrap();
    for func in &module.functions {
        out.push_str(&print_function(func));
    }
    writeln!(&mut out, "}}").unwrap();
    out
}

pub fn print_function(func: &Function) -> String {
    let mut out = String::new();

    // ── Signature ───────────────────────────────────────────────────
    out.push_str("  func.func @");
    out.push_str(&func.name);
    out.push('(');

    for (i, p) in func.params.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write!(&mut out, "{}: {}", p.value, p.shape.mlir_tensor_type()).unwrap();
    }
    out.push(')');

    // Return type(s)
    let ret_types: Vec<String> = func
        .returns
        .iter()
        .filter_map(|&rid| value_type(func, rid).map(|s| s.mlir_tensor_type()))
        .collect();

    match ret_types.len() {
        0 => {}
        1 => write!(&mut out, " -> {}", ret_types[0]).unwrap(),
        _ => write!(&mut out, " -> ({})", ret_types.join(", ")).unwrap(),
    }

    out.push_str(" {\n");

    // ── Body ────────────────────────────────────────────────────────
    for stmt in &func.insts {
        match &stmt.inst {
            Inst::DotGeneral(op) => {
                let lhs_ty = value_type(func, op.lhs).unwrap().mlir_tensor_type();
                let rhs_ty = value_type(func, op.rhs).unwrap().mlir_tensor_type();
                let out_ty = op.out.mlir_tensor_type();
                writeln!(
                    &mut out,
                    "    {} = \"stablehlo.dot_general\"({}, {}) {{ dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [{}], rhs_batching_dimensions = [{}], lhs_contracting_dimensions = [{}], rhs_contracting_dimensions = [{}]> }} : ({}, {}) -> {}",
                    stmt.result, op.lhs, op.rhs,
                    join_i64(&op.batching_dims_lhs),
                    join_i64(&op.batching_dims_rhs),
                    join_i64(&op.contracting_dims_lhs),
                    join_i64(&op.contracting_dims_rhs),
                    lhs_ty, rhs_ty, out_ty
                ).unwrap();
            }

            Inst::BroadcastInDim(op) => {
                let in_ty = value_type(func, op.operand).unwrap().mlir_tensor_type();
                let out_ty = op.out.mlir_tensor_type();
                writeln!(
                    &mut out,
                    "    {} = stablehlo.broadcast_in_dim {}, dims = [{}] : ({}) -> {}",
                    stmt.result,
                    op.operand,
                    join_i64(&op.dims),
                    in_ty,
                    out_ty
                )
                .unwrap();
            }

            // Binary ops
            Inst::Add(op) => print_binary(&mut out, "add", stmt.result, op),
            Inst::Subtract(op) => print_binary(&mut out, "subtract", stmt.result, op),
            Inst::Multiply(op) => print_binary(&mut out, "multiply", stmt.result, op),
            Inst::Divide(op) => print_binary(&mut out, "divide", stmt.result, op),
            Inst::Maximum(op) => print_binary(&mut out, "maximum", stmt.result, op),

            // Unary ops
            Inst::Negate(op) => print_unary(&mut out, "negate", stmt.result, op, func),
            Inst::Exponential(op) => print_unary(&mut out, "exponential", stmt.result, op, func),
            Inst::Log(op) => print_unary(&mut out, "log", stmt.result, op, func),
            Inst::Sqrt(op) => print_unary(&mut out, "sqrt", stmt.result, op, func),
            Inst::Rsqrt(op) => print_unary(&mut out, "rsqrt", stmt.result, op, func),
            Inst::Abs(op) => print_unary(&mut out, "abs", stmt.result, op, func),
            Inst::Tanh(op) => print_unary(&mut out, "tanh", stmt.result, op, func),
            Inst::Logistic(op) => print_unary(&mut out, "logistic", stmt.result, op, func),
            Inst::Cosine(op) => print_unary(&mut out, "cosine", stmt.result, op, func),
            Inst::Sine(op) => print_unary(&mut out, "sine", stmt.result, op, func),

            Inst::Convert(op) => {
                let in_ty = value_type(func, op.operand).unwrap().mlir_tensor_type();
                let out_ty = op.out.mlir_tensor_type();
                writeln!(
                    &mut out,
                    "    {} = stablehlo.convert {} : ({}) -> {}",
                    stmt.result, op.operand, in_ty, out_ty
                )
                .unwrap();
            }

            Inst::Constant(op) => {
                let out_ty = op.out.mlir_tensor_type();
                let val_str = if op.out.dtype.is_float() {
                    format_mlir_float(op.value)
                } else {
                    format!("{}", op.value as i64)
                };
                writeln!(
                    &mut out,
                    "    {} = stablehlo.constant dense<{}> : {}",
                    stmt.result, val_str, out_ty
                )
                .unwrap();
            }

            Inst::Reshape(op) => {
                let in_ty = value_type(func, op.operand).unwrap().mlir_tensor_type();
                let out_ty = op.out.mlir_tensor_type();
                writeln!(
                    &mut out,
                    "    {} = stablehlo.reshape {} : ({}) -> {}",
                    stmt.result, op.operand, in_ty, out_ty
                )
                .unwrap();
            }

            Inst::Transpose(op) => {
                let in_ty = value_type(func, op.operand).unwrap().mlir_tensor_type();
                let out_ty = op.out.mlir_tensor_type();
                writeln!(
                    &mut out,
                    "    {} = \"stablehlo.transpose\"({}) {{permutation = array<i64: {}>}} : ({}) -> {}",
                    stmt.result, op.operand,
                    join_i64(&op.permutation),
                    in_ty, out_ty
                ).unwrap();
            }

            Inst::Reduce(op) => {
                let operand_ty = value_type(func, op.operand).unwrap().mlir_tensor_type();
                let init_ty = value_type(func, op.init_value).unwrap().mlir_tensor_type();
                let out_ty = op.out.mlir_tensor_type();
                let scalar_ty = Shape::new(vec![], op.out.dtype).mlir_tensor_type();

                let body_op = match op.kind {
                    ReduceKind::Sum => "stablehlo.add",
                    ReduceKind::Max => "stablehlo.maximum",
                    ReduceKind::Min => "stablehlo.minimum",
                };

                writeln!(
                    &mut out,
                    "    {} = \"stablehlo.reduce\"({}, {}) ({{",
                    stmt.result, op.operand, op.init_value
                )
                .unwrap();
                writeln!(
                    &mut out,
                    "    ^bb0(%reduce_arg0: {scalar_ty}, %reduce_arg1: {scalar_ty}):"
                )
                .unwrap();
                writeln!(
                    &mut out,
                    "      %reduce_result_{} = {body_op} %reduce_arg0, %reduce_arg1 : {scalar_ty}",
                    stmt.result.0
                )
                .unwrap();
                writeln!(
                    &mut out,
                    "      stablehlo.return %reduce_result_{} : {scalar_ty}",
                    stmt.result.0
                )
                .unwrap();
                writeln!(
                    &mut out,
                    "    }}) {{\n      dimensions = array<i64: {}>\n    }} : ({}, {}) -> {}",
                    join_i64(&op.dimensions),
                    operand_ty,
                    init_ty,
                    out_ty
                )
                .unwrap();
            }

            Inst::ReduceArgMax(op) => {
                let val_ty = value_type(func, op.operand).unwrap().mlir_tensor_type();
                let idx_ty = value_type(func, op.iota).unwrap().mlir_tensor_type();
                let init_val_ty = value_type(func, op.init_value).unwrap().mlir_tensor_type();
                let init_idx_ty = value_type(func, op.init_index).unwrap().mlir_tensor_type();
                let operand_dtype = value_type(func, op.operand).unwrap().dtype;
                let scalar_val_ty = Shape::new(vec![], operand_dtype).mlir_tensor_type();
                let scalar_idx_ty = Shape::new(vec![], DType::I32).mlir_tensor_type();

                let mut out_val_dims = Vec::new();
                let operand_shape = value_type(func, op.operand).unwrap();
                for (i, &d) in operand_shape.dims.iter().enumerate() {
                    if i as i64 != op.dimension {
                        out_val_dims.push(d);
                    }
                }
                let out_val_ty = Shape::new(out_val_dims, operand_dtype).mlir_tensor_type();
                let out_idx_ty = op.out.mlir_tensor_type();

                let r = stmt.result.0;
                writeln!(
                    &mut out,
                    "    %argmax_val_{r}, {} = \"stablehlo.reduce\"({}, {}, {}, {}) ({{",
                    stmt.result, op.operand, op.iota, op.init_value, op.init_index
                )
                .unwrap();
                writeln!(
                    &mut out,
                    "    ^bb0(%arg_val_0: {scalar_val_ty}, %arg_idx_0: {scalar_idx_ty}, %arg_val_1: {scalar_val_ty}, %arg_idx_1: {scalar_idx_ty}):"
                )
                .unwrap();
                writeln!(
                    &mut out,
                    "      %cmp_{r} = stablehlo.compare GE, %arg_val_0, %arg_val_1, FLOAT : ({scalar_val_ty}, {scalar_val_ty}) -> tensor<i1>"
                )
                .unwrap();
                writeln!(
                    &mut out,
                    "      %sel_val_{r} = stablehlo.select %cmp_{r}, %arg_val_0, %arg_val_1 : (tensor<i1>, {scalar_val_ty}, {scalar_val_ty}) -> {scalar_val_ty}"
                )
                .unwrap();
                writeln!(
                    &mut out,
                    "      %sel_idx_{r} = stablehlo.select %cmp_{r}, %arg_idx_0, %arg_idx_1 : (tensor<i1>, {scalar_idx_ty}, {scalar_idx_ty}) -> {scalar_idx_ty}"
                )
                .unwrap();
                writeln!(
                    &mut out,
                    "      stablehlo.return %sel_val_{r}, %sel_idx_{r} : {scalar_val_ty}, {scalar_idx_ty}"
                )
                .unwrap();
                writeln!(
                    &mut out,
                    "    }}) {{\n      dimensions = array<i64: {}>\n    }} : ({}, {}, {}, {}) -> ({}, {})",
                    op.dimension,
                    val_ty, idx_ty, init_val_ty, init_idx_ty,
                    out_val_ty, out_idx_ty
                )
                .unwrap();
            }

            Inst::Concatenate(op) => {
                let operand_strs: Vec<String> =
                    op.operands.iter().map(|v| format!("{}", v)).collect();
                let operand_types: Vec<String> = op
                    .operands
                    .iter()
                    .map(|&v| value_type(func, v).unwrap().mlir_tensor_type())
                    .collect();
                let out_ty = op.out.mlir_tensor_type();
                writeln!(
                    &mut out,
                    "    {} = stablehlo.concatenate {}, dim = {} : ({}) -> {}",
                    stmt.result,
                    operand_strs.join(", "),
                    op.dimension,
                    operand_types.join(", "),
                    out_ty
                )
                .unwrap();
            }

            Inst::Slice(op) => {
                let in_ty = value_type(func, op.operand).unwrap().mlir_tensor_type();
                let out_ty = op.out.mlir_tensor_type();
                let ranges: Vec<String> = op
                    .start_indices
                    .iter()
                    .zip(op.limit_indices.iter())
                    .zip(op.strides.iter())
                    .map(|((&s, &l), &st)| {
                        if st == 1 {
                            format!("{s}:{l}")
                        } else {
                            format!("{s}:{l}:{st}")
                        }
                    })
                    .collect();
                writeln!(
                    &mut out,
                    "    {} = stablehlo.slice {} [{}] : ({}) -> {}",
                    stmt.result,
                    op.operand,
                    ranges.join(", "),
                    in_ty,
                    out_ty
                )
                .unwrap();
            }

            Inst::Compare(op) => {
                let lhs_shape = value_type(func, op.lhs).unwrap();
                let compare_type = if lhs_shape.dtype.is_float() {
                    "FLOAT"
                } else {
                    "SIGNED"
                };
                let lhs_ty = lhs_shape.mlir_tensor_type();
                let rhs_ty = value_type(func, op.rhs).unwrap().mlir_tensor_type();
                let out_ty = op.out.mlir_tensor_type();
                writeln!(
                    &mut out,
                    "    {} = stablehlo.compare {}, {}, {}, {} : ({}, {}) -> {}",
                    stmt.result,
                    op.direction.mlir_str(),
                    op.lhs,
                    op.rhs,
                    compare_type,
                    lhs_ty,
                    rhs_ty,
                    out_ty
                )
                .unwrap();
            }

            Inst::Select(op) => {
                let pred_ty = value_type(func, op.pred).unwrap().mlir_tensor_type();
                let true_ty = value_type(func, op.on_true).unwrap().mlir_tensor_type();
                let false_ty = value_type(func, op.on_false).unwrap().mlir_tensor_type();
                let out_ty = op.out.mlir_tensor_type();
                writeln!(
                    &mut out,
                    "    {} = stablehlo.select {}, {}, {} : ({}, {}, {}) -> {}",
                    stmt.result, op.pred, op.on_true, op.on_false, pred_ty, true_ty, false_ty, out_ty
                )
                .unwrap();
            }

            Inst::Iota(op) => {
                let out_ty = op.out.mlir_tensor_type();
                writeln!(
                    &mut out,
                    "    {} = stablehlo.iota dim = {} : {}",
                    stmt.result, op.iota_dimension, out_ty
                )
                .unwrap();
            }

            Inst::Gather(op) => {
                let operand_ty = value_type(func, op.operand).unwrap().mlir_tensor_type();
                let indices_ty = value_type(func, op.start_indices)
                    .unwrap()
                    .mlir_tensor_type();
                let out_ty = op.out.mlir_tensor_type();
                writeln!(
                    &mut out,
                    "    {} = \"stablehlo.gather\"({}, {}) {{dimension_numbers = #stablehlo.gather<offset_dims = [{}], collapsed_slice_dims = [{}], start_index_map = [{}], index_vector_dim = {}>, slice_sizes = array<i64: {}>}} : ({}, {}) -> {}",
                    stmt.result,
                    op.operand,
                    op.start_indices,
                    join_i64(&op.offset_dims),
                    join_i64(&op.collapsed_slice_dims),
                    join_i64(&op.start_index_map),
                    op.index_vector_dim,
                    join_i64(&op.slice_sizes),
                    operand_ty,
                    indices_ty,
                    out_ty
                ).unwrap();
            }
        }
    }

    // ── Return ──────────────────────────────────────────────────────
    if func.returns.is_empty() {
        out.push_str("    // NOTE: no return set\n");
    } else {
        let ret_strs: Vec<String> = func.returns.iter().map(|&r| format!("{}", r)).collect();
        let ret_tys: Vec<String> = func
            .returns
            .iter()
            .map(|&r| value_type(func, r).unwrap().mlir_tensor_type())
            .collect();
        writeln!(
            &mut out,
            "    return {} : {}",
            ret_strs.join(", "),
            ret_tys.join(", ")
        )
        .unwrap();
    }

    out.push_str("  }\n");
    out
}

// ── Helpers ─────────────────────────────────────────────────────────

fn print_binary(out: &mut String, name: &str, result: ValueId, op: &crate::ir::BinaryOp) {
    let out_ty = op.out.mlir_tensor_type();
    writeln!(
        out,
        "    {} = stablehlo.{} {}, {} : {}",
        result, name, op.lhs, op.rhs, out_ty
    )
    .unwrap();
}

fn print_unary(
    out: &mut String,
    name: &str,
    result: ValueId,
    op: &crate::ir::UnaryOp,
    func: &Function,
) {
    let in_ty = value_type(func, op.operand).unwrap().mlir_tensor_type();
    writeln!(
        out,
        "    {} = stablehlo.{} {} : {}",
        result, name, op.operand, in_ty
    )
    .unwrap();
}

/// Format a float the way MLIR expects: `1.000000e+00`, or hex for special values.
fn format_mlir_float(v: f64) -> String {
    if v.is_nan() {
        return "0x7FC00000".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 {
            "0x7F800000".to_string()
        } else {
            "0xFF800000".to_string()
        };
    }
    // Rust's {:e} doesn't zero-pad exponents or add '+' sign on them.
    // MLIR expects the form: [-]d.ddddddE[+-]dd  (two-digit exponent with sign)
    let s = format!("{:.6e}", v);
    // s looks like "1.000000e5" or "-1.000000e-6" or "0.000000e0"
    // We need to normalize the exponent to "+05" / "-06" / "+00"
    if let Some(pos) = s.find('e') {
        let (mantissa, exp_part) = s.split_at(pos);
        let exp_str = &exp_part[1..]; // skip the 'e'
        let exp_val: i32 = exp_str.parse().unwrap();
        format!("{mantissa}e{exp_val:+03}")
    } else {
        s
    }
}

fn join_i64(xs: &[i64]) -> String {
    xs.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn value_type(func: &Function, id: ValueId) -> Option<&Shape> {
    for p in &func.params {
        if p.value == id {
            return Some(&p.shape);
        }
    }
    for stmt in &func.insts {
        if stmt.result == id {
            return Some(inst_out_shape(&stmt.inst));
        }
    }
    None
}

fn inst_out_shape(inst: &Inst) -> &Shape {
    match inst {
        Inst::DotGeneral(op) => &op.out,
        Inst::BroadcastInDim(op) => &op.out,
        Inst::Add(op)
        | Inst::Subtract(op)
        | Inst::Multiply(op)
        | Inst::Divide(op)
        | Inst::Maximum(op) => &op.out,
        Inst::Negate(op)
        | Inst::Exponential(op)
        | Inst::Log(op)
        | Inst::Sqrt(op)
        | Inst::Rsqrt(op)
        | Inst::Abs(op)
        | Inst::Tanh(op)
        | Inst::Logistic(op)
        | Inst::Cosine(op)
        | Inst::Sine(op)
        | Inst::Convert(op)
        | Inst::Reshape(op) => &op.out,
        Inst::Constant(op) => &op.out,
        Inst::Transpose(op) => &op.out,
        Inst::Reduce(op) => &op.out,
        Inst::ReduceArgMax(op) => &op.out,
        Inst::Concatenate(op) => &op.out,
        Inst::Slice(op) => &op.out,
        Inst::Compare(op) => &op.out,
        Inst::Select(op) => &op.out,
        Inst::Iota(op) => &op.out,
        Inst::Gather(op) => &op.out,
    }
}
