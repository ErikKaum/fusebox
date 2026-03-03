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
    let func = trace_function(name, |cx| Ok(f(cx))).unwrap();
    let module = Module {
        functions: vec![func],
    };
    print_module(&module)
}

fn main() {
    let fixtures: Vec<(&str, String)> = vec![
        (
            "binary_broadcast.mlir",
            trace_to_mlir("binary_broadcast", |cx| {
                let a = cx.input("a", f32_shape(&[3, 4]));
                let b = cx.input("b", f32_shape(&[4]));
                a.add(&b).unwrap()
            }),
        ),
        (
            "unary_chain.mlir",
            trace_to_mlir("unary_chain", |cx| {
                let a = cx.input("a", f32_shape(&[2, 3]));
                a.exp().log().tanh()
            }),
        ),
        (
            "matmul_batched.mlir",
            trace_to_mlir("matmul_batched", |cx| {
                let a = cx.input("a", f32_shape(&[2, 3, 4]));
                let b = cx.input("b", f32_shape(&[4, 5]));
                a.matmul(&b).unwrap()
            }),
        ),
        (
            "reduce_keepdim.mlir",
            trace_to_mlir("reduce_keepdim", |cx| {
                let a = cx.input("a", f32_shape(&[3, 4, 5]));
                a.sum_keepdim(&[1]).unwrap()
            }),
        ),
        (
            "softmax.mlir",
            trace_to_mlir("softmax", |cx| {
                let a = cx.input("a", f32_shape(&[2, 4]));
                a.softmax(-1).unwrap()
            }),
        ),
        (
            "concat_slice.mlir",
            trace_to_mlir("concat_slice", |cx| {
                let a = cx.input("a", f32_shape(&[2, 4]));
                let b = cx.input("b", f32_shape(&[3, 4]));
                let cat = Tensor::cat(&[&a, &b], 0).unwrap();
                cat.narrow(0, 1, 3).unwrap()
            }),
        ),
        (
            "compare_select.mlir",
            trace_to_mlir("compare_select", |cx| {
                let a = cx.input("a", f32_shape(&[3, 4]));
                let b = cx.input("b", f32_shape(&[3, 4]));
                let zero = a.zeros_like();
                let mask = a.gt(&b).unwrap();
                Tensor::select(&mask, &a, &zero).unwrap()
            }),
        ),
        (
            "scalar_ops.mlir",
            trace_to_mlir("scalar_ops", |cx| {
                let a = cx.input("a", f32_shape(&[3, 4]));
                let b = (&a * 2.0).unwrap();
                (&b + 1.0).unwrap()
            }),
        ),
        ("iota.mlir", {
            let func = trace_function("iota_test", |cx| {
                cx.iota(Shape::new(vec![4, 4], DType::I32), 0)
            })
            .unwrap();
            let module = Module {
                functions: vec![func],
            };
            print_module(&module)
        }),
        (
            "gather.mlir",
            trace_to_mlir("gather_test", |cx| {
                let table = cx.input("table", f32_shape(&[100, 32]));
                let indices = cx.input("idx", Shape::new(vec![8], DType::I32));
                table.gather(&indices).unwrap()
            }),
        ),
        (
            "argmax.mlir",
            trace_to_mlir("argmax_test", |cx| {
                let a = cx.input("a", f32_shape(&[2, 3, 4]));
                a.argmax(-1).unwrap()
            }),
        ),
        (
            "cos_sin.mlir",
            trace_to_mlir("cos_sin_test", |cx| {
                let a = cx.input("a", f32_shape(&[4, 8]));
                let c = a.cos();
                let s = a.sin();
                (&c + &s).unwrap()
            }),
        ),
        ("convert.mlir", {
            let func = trace_function("convert_test", |cx| {
                let a = cx.iota(Shape::new(vec![4], DType::I32), 0)?;
                Ok(a.to_dtype(DType::F32))
            })
            .unwrap();
            let module = Module {
                functions: vec![func],
            };
            print_module(&module)
        }),
    ];

    let dir = std::path::Path::new("tests/fixtures");
    std::fs::create_dir_all(dir).unwrap();

    for (name, mlir) in &fixtures {
        let path = dir.join(name);
        std::fs::write(&path, mlir).unwrap();
        println!("wrote {}", path.display());
    }

    println!("\n{} fixtures generated.", fixtures.len());
}
