use fusebox::checkpoint;
use fusebox::dtype::DType;
use fusebox::engine::Engine;
use fusebox::error::Error;
use fusebox::module_api::Module;
use fusebox::pjrt_runtime::default_cpu_plugin_path;
use fusebox::shape::Shape;
use fusebox::tensor::Tensor;
use fusebox::trace::TraceCx;
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
    let ckpt = checkpoint::Checkpoint::from_file("model.safetensors")?;
    let engine = Engine::new(default_cpu_plugin_path());

    let runner = engine.compile("main", |cx| {
        let x = cx.input("x", Shape::new(vec![2, 4], DType::F32));
        let mlp = Mlp::trace(cx, "proj", ckpt.shapes())?;
        mlp.forward(cx, &x)
    })?;

    let weights = ckpt.weights_f32_for_signature(runner.signature())?;
    let sess = runner.session_f32(weights);

    let y = sess.run(|inputs| {
        inputs.set_input("x", vec![1., 2., 3., 4., 5., 6., 7., 8.])
    })?;

    println!("{:?}", y.dims);
    println!("{:?}", y.data);
    Ok(())
}