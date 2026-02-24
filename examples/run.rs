use fusebox::{
    builder::FuncBuilder,
    dtype::DType,
    ir::Module,
    pjrt_runtime::{HostTensorF32, PjrtCpuRunner, default_cpu_plugin_path},
    print_mlir::print_module,
    shape::Shape,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let b: i64 = 2; // batch
    let inn: i64 = 4; // input dim
    let out: i64 = 3; // output dim

    let mut f = FuncBuilder::new("main");

    // Parameters (these are symbolic; no data)
    let x = f.param("x", Shape::new(vec![b, inn], DType::F32)); // [B, In]
    let w = f.param("w", Shape::new(vec![inn, out], DType::F32)); // [In, Out]
    let bias = f.param("b", Shape::new(vec![out], DType::F32)); // [Out]

    // y = x·w
    let y = f.matmul_2d(&x, &w)?;

    // bb = broadcast(bias) to [B, Out]
    let bb = f.broadcast_bias_1d(&bias, b)?;

    // z = y + bb
    let z = f.add(&y, &bb)?;

    f.ret(&z);

    let func = f.finish();
    let module = Module {
        functions: vec![func],
    };

    println!("{}", print_module(&module));

    let mlir = print_module(&module); // your existing print_mlir output

    let runner = PjrtCpuRunner::from_mlir_text(&mlir, default_cpu_plugin_path())?;

    let x = HostTensorF32::new(vec![2, 4], vec![1., 2., 3., 4., 5., 6., 7., 8.])?;
    let w = HostTensorF32::new(
        vec![4, 3],
        vec![1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.],
    )?;
    let b = HostTensorF32::new(vec![3], vec![10., 20., 30.])?;

    let y = runner.run_f32(vec![x, w, b])?;
    println!("output dims={:?}", y.dims);
    println!("output flat={:?}", y.data);

    Ok(())
}
