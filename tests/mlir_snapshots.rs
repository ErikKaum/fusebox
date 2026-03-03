use fusebox::dtype::DType;
use fusebox::ir::Module;
use fusebox::print_mlir::print_module;
use fusebox::shape::Shape;
use fusebox::tensor::Tensor;
use fusebox::trace::TraceCx;
use fusebox::trace_fn::trace_function;

fn f32_shape(dims: &[i64]) -> Shape {
    Shape::new(dims.to_vec(), DType::F32)
}

fn trace_to_mlir(name: &str, f: impl FnOnce(&mut TraceCx) -> Tensor) -> String {
    let func = trace_function(name, |cx| {
        let out = f(cx);
        Ok(out)
    })
    .unwrap();
    let module = Module {
        functions: vec![func],
    };
    print_module(&module)
}

fn assert_mlir_fixture(fixture_name: &str, actual: &str) {
    let fixture_path = format!("tests/fixtures/{}", fixture_name);
    let expected = std::fs::read_to_string(&fixture_path).unwrap_or_else(|e| {
        panic!(
            "Cannot read fixture {}: {}\n\
             Regenerate fixtures with: just update-fixtures\n\
             Actual MLIR:\n{}",
            fixture_path, e, actual
        )
    });
    assert_eq!(
        actual.trim(),
        expected.trim(),
        "\nMLIR mismatch for fixture: {}\n\
         Regenerate fixtures with: just update-fixtures",
        fixture_name
    );
}

#[test]
fn binary_broadcast() {
    let mlir = trace_to_mlir("binary_broadcast", |cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = cx.input("b", f32_shape(&[4]));
        a.add(&b).unwrap()
    });
    assert_mlir_fixture("binary_broadcast.mlir", &mlir);
}

#[test]
fn unary_chain() {
    let mlir = trace_to_mlir("unary_chain", |cx| {
        let a = cx.input("a", f32_shape(&[2, 3]));
        a.exp().log().tanh()
    });
    assert_mlir_fixture("unary_chain.mlir", &mlir);
}

#[test]
fn matmul_batched() {
    let mlir = trace_to_mlir("matmul_batched", |cx| {
        let a = cx.input("a", f32_shape(&[2, 3, 4]));
        let b = cx.input("b", f32_shape(&[4, 5]));
        a.matmul(&b).unwrap()
    });
    assert_mlir_fixture("matmul_batched.mlir", &mlir);
}

#[test]
fn reduce_keepdim() {
    let mlir = trace_to_mlir("reduce_keepdim", |cx| {
        let a = cx.input("a", f32_shape(&[3, 4, 5]));
        a.sum_keepdim(&[1]).unwrap()
    });
    assert_mlir_fixture("reduce_keepdim.mlir", &mlir);
}

#[test]
fn softmax_graph() {
    let mlir = trace_to_mlir("softmax", |cx| {
        let a = cx.input("a", f32_shape(&[2, 4]));
        a.softmax(-1).unwrap()
    });
    assert_mlir_fixture("softmax.mlir", &mlir);
}

#[test]
fn concat_slice() {
    let mlir = trace_to_mlir("concat_slice", |cx| {
        let a = cx.input("a", f32_shape(&[2, 4]));
        let b = cx.input("b", f32_shape(&[3, 4]));
        let cat = Tensor::cat(&[&a, &b], 0).unwrap();
        cat.narrow(0, 1, 3).unwrap()
    });
    assert_mlir_fixture("concat_slice.mlir", &mlir);
}

#[test]
fn compare_select() {
    let mlir = trace_to_mlir("compare_select", |cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = cx.input("b", f32_shape(&[3, 4]));
        let zero = a.zeros_like();
        let mask = a.gt(&b).unwrap();
        Tensor::select(&mask, &a, &zero).unwrap()
    });
    assert_mlir_fixture("compare_select.mlir", &mlir);
}

#[test]
fn scalar_ops() {
    let mlir = trace_to_mlir("scalar_ops", |cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = (&a * 2.0).unwrap();
        (&b + 1.0).unwrap()
    });
    assert_mlir_fixture("scalar_ops.mlir", &mlir);
}

#[test]
fn iota_graph() {
    let func = trace_function("iota_test", |cx| {
        let idx = cx.iota(Shape::new(vec![4, 4], DType::I32), 0)?;
        Ok(idx)
    })
    .unwrap();
    let module = Module {
        functions: vec![func],
    };
    let mlir = print_module(&module);
    assert_mlir_fixture("iota.mlir", &mlir);
}

#[test]
fn gather_graph() {
    let mlir = trace_to_mlir("gather_test", |cx| {
        let table = cx.input("table", f32_shape(&[100, 32]));
        let indices = cx.input("idx", Shape::new(vec![8], DType::I32));
        table.gather(&indices).unwrap()
    });
    assert_mlir_fixture("gather.mlir", &mlir);
}

#[test]
fn argmax_graph() {
    let mlir = trace_to_mlir("argmax_test", |cx| {
        let a = cx.input("a", f32_shape(&[2, 3, 4]));
        a.argmax(-1).unwrap()
    });
    assert_mlir_fixture("argmax.mlir", &mlir);
}

#[test]
fn cos_sin_graph() {
    let mlir = trace_to_mlir("cos_sin_test", |cx| {
        let a = cx.input("a", f32_shape(&[4, 8]));
        let c = a.cos();
        let s = a.sin();
        (&c + &s).unwrap()
    });
    assert_mlir_fixture("cos_sin.mlir", &mlir);
}

#[test]
fn convert_graph() {
    let func = trace_function("convert_test", |cx| {
        let a = cx.iota(Shape::new(vec![4], DType::I32), 0)?;
        Ok(a.to_dtype(DType::F32))
    })
    .unwrap();
    let module = Module {
        functions: vec![func],
    };
    let mlir = print_module(&module);
    assert_mlir_fixture("convert.mlir", &mlir);
}
