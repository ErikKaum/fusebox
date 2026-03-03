use fusebox::prelude::*;

#[derive(Module)]
pub struct Mlp {
    pub up: Linear,
    #[module(name = "gate_proj")]
    pub gate: Linear,
    pub down: Linear,
}

impl Mlp {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let gate = self.gate.forward(x)?;
        let gate = gate.silu();
        let up = self.up.forward(x)?;
        let hidden = (&gate * &up)?;
        self.down.forward(&hidden)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ckpt = Checkpoint::from_file("examples/linear/artifacts/model.safetensors")?;
    let device = Device::cpu();

    let runner = device.compile("main", |cx| {
        let x = cx.input("x", Shape::new(vec![1, 8], DType::F32));
        let mlp = Mlp::trace(cx, "proj", ckpt.shapes())?;
        mlp.forward(&x)
    })?;

    let weights = ckpt.load_weights(runner.signature())?;
    let sess = runner.session(weights);

    let y = sess.run(|inputs| inputs.set_input("x", vec![1., 2., 3., 4., 5., 6., 7., 8.]))?;

    println!("shape: {:?}", y.shape());
    println!("data: {:?}", y.to_f32().unwrap());
    Ok(())
}
