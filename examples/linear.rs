use fusebox::builder::FuncBuilder;
use fusebox::dtype::DType;
use fusebox::ir::Module;
use fusebox::print_mlir::print_module;
use fusebox::shape::Shape;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Hardcode some dimensions for the demo.
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
    Ok(())
}
