// src/nn/linear.rs
use fusebox::dtype::DType;
use fusebox::error::Error;
use fusebox::module_api::Module;
use fusebox::pjrt_runtime::{PjrtCpuRunner, default_cpu_plugin_path};
use fusebox::safetensor_shapes::SafeTensorShapes;
use fusebox::shape::Shape;
use fusebox::tensor::Tensor;
use fusebox::trace::TraceCx;
use fusebox::weights::WeightsF32;
use fusebox_macros::Module;

#[derive(Module)]
pub struct Linear {
    pub w: Tensor,
    pub b: Option<Tensor>,
}

impl Linear {
    pub fn forward(&self, cx: &mut TraceCx, x: &Tensor) -> Result<Tensor, Error> {
        let y = cx.matmul_2d(x, &self.w)?;
        if let Some(b) = &self.b {
            let batch = x.shape.dim(0);
            let bb = cx.broadcast_bias_1d(b, batch)?;
            cx.add(&y, &bb)
        } else {
            Ok(y)
        }
    }
}

#[derive(Module)]
pub struct Mlp {
    pub up: Linear,
    #[module(name = "gate_proj")]
    pub gate: Linear,
    pub down: Linear,
}

impl Mlp {
    pub fn forward(&self, cx: &mut TraceCx, x: &Tensor) -> Result<Tensor, Error> {
        let gate = self.gate.forward(cx, x)?;
        let gate = cx.silu(&gate)?;
        let up = self.up.forward(cx, x)?;
        let hidden = cx.mul(&gate, &up)?;
        self.down.forward(cx, &hidden)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cx = TraceCx::new("main");

    let x = cx.input("x", Shape::new(vec![2, 4], DType::F32));

    let shapes = SafeTensorShapes::from_file("model.safetensors").unwrap();
    let mlp = Mlp::trace(&mut cx, "proj", &shapes).unwrap();
    let y = mlp.forward(&mut cx, &x)?;

    cx.ret(&y);

    let func = cx.finish();

    let runner = PjrtCpuRunner::from_function(&func, default_cpu_plugin_path())?;

    // load weights for every param except runtime inputs
    let weights =
        WeightsF32::from_safetensors_for_weights("model.safetensors", runner.signature())?;

    // build bindings from signature, apply weights, then set runtime input(s)
    let mut bindings = runner.inputs_f32();
    weights.apply_into(&mut bindings)?;
    bindings.set("x", vec![1., 2., 3., 4., 5., 6., 7., 8.])?;

    let y = runner.run_f32_inputs(bindings)?;

    println!("{:?}", y.dims);
    println!("{:?}", y.data);

    Ok(())
}
