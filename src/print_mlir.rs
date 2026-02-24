// turns our IR into MLIR text that looks like StableHLO. All syntax lives here so the rest of the code stays stable

use core::fmt::Write;

use crate::ir::{Function, Inst, Module};
use crate::shape::Shape;
use crate::value::ValueId;

/// Print a whole module as MLIR text.
pub fn print_module(module: &Module) -> String {
    let mut out = String::new();
    writeln!(&mut out, "module {{").unwrap();

    for func in &module.functions {
        out.push_str(&print_function(func));
        out.push('\n');
    }

    writeln!(&mut out, "}}").unwrap();
    out
}

/// Print one function as MLIR text (func.func + stablehlo ops).
pub fn print_function(func: &Function) -> String {
    let mut out = String::new();

    // Signature
    out.push_str("  func.func @");
    out.push_str(&func.name);
    out.push('(');

    for (i, p) in func.params.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        // MLIR args are SSA values too: %0: tensor<...>
        write!(&mut out, "{}: {}", p.value, p.shape.mlir_tensor_type()).unwrap();
    }

    out.push(')');

    // Return type (optional, but nice for clarity)
    let ret_ty = func
        .ret
        .and_then(|rid| value_type(func, rid))
        .map(|s| s.mlir_tensor_type());

    if let Some(ret_ty) = ret_ty {
        write!(&mut out, " -> {}", ret_ty).unwrap();
    }

    out.push_str(" {\n");

    // Body
    for stmt in &func.insts {
        match &stmt.inst {
            Inst::DotGeneral(op) => {
                // %r = stablehlo.dot_general %lhs, %rhs, ... : (ty, ty) -> ty
                let lhs_ty = value_type(func, op.lhs).unwrap().mlir_tensor_type();
                let rhs_ty = value_type(func, op.rhs).unwrap().mlir_tensor_type();
                let out_ty = op.out.mlir_tensor_type();

                write!(
                    &mut out,
                    "    {} = \"stablehlo.dot_general\"({}, {}) {{ dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [{}], rhs_batching_dimensions = [{}], lhs_contracting_dimensions = [{}], rhs_contracting_dimensions = [{}]> }} : ({}, {}) -> {}\n",
                    stmt.result,
                    op.lhs,
                    op.rhs,
                    join_i64(&op.batching_dims_lhs),
                    join_i64(&op.batching_dims_rhs),
                    join_i64(&op.contracting_dims_lhs),
                    join_i64(&op.contracting_dims_rhs),
                    lhs_ty,
                    rhs_ty,
                    out_ty
                ).unwrap();
            }

            Inst::BroadcastInDim(op) => {
                // %r = stablehlo.broadcast_in_dim %operand, dims = [...] : (in_ty) -> out_ty
                let in_ty = value_type(func, op.operand).unwrap().mlir_tensor_type();
                let out_ty = op.out.mlir_tensor_type();
                write!(
                    &mut out,
                    "    {} = stablehlo.broadcast_in_dim {}, dims = [{}] : ({}) -> {}\n",
                    stmt.result,
                    op.operand,
                    join_i64(&op.dims),
                    in_ty,
                    out_ty
                )
                .unwrap();
            }

            Inst::Add(op) => {
                // %r = stablehlo.add %lhs, %rhs : out_ty
                let out_ty = op.out.mlir_tensor_type();
                write!(
                    &mut out,
                    "    {} = stablehlo.add {}, {} : {}\n",
                    stmt.result, op.lhs, op.rhs, out_ty
                )
                .unwrap();
            }
        }
    }

    // Return
    if let Some(ret) = func.ret {
        let ret_ty = value_type(func, ret).unwrap().mlir_tensor_type();
        write!(&mut out, "    return {} : {}\n", ret, ret_ty).unwrap();
    } else {
        out.push_str("    // NOTE: no return set\n");
    }

    out.push_str("  }\n");
    out
}

fn join_i64(xs: &[i64]) -> String {
    xs.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Find the shape/type of a value by scanning params and instructions.
/// This is O(n) and fine for now.
fn value_type(func: &Function, id: ValueId) -> Option<&Shape> {
    for p in &func.params {
        if p.value == id {
            return Some(&p.shape);
        }
    }
    for stmt in &func.insts {
        if stmt.result == id {
            return match &stmt.inst {
                Inst::DotGeneral(op) => Some(&op.out),
                Inst::BroadcastInDim(op) => Some(&op.out),
                Inst::Add(op) => Some(&op.out),
            };
        }
    }
    None
}
