use fusebox::prelude::*;

const N_HEADS: i64 = 2;
const HEAD_DIM: i64 = 8;

// ── Multi-Head Attention ────────────────────────────────────────────

#[derive(Module)]
pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
}

impl Attention {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let [batch, seq, _dim] = x.shape.dims[..] else {
            return Err(Error::Unsupported {
                op: "attention",
                msg: "expected rank-3 input [batch, seq, dim]",
            });
        };

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape(&[batch, seq, N_HEADS, HEAD_DIM])?
            .transpose(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[batch, seq, N_HEADS, HEAD_DIM])?
            .transpose(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[batch, seq, N_HEADS, HEAD_DIM])?
            .transpose(&[0, 2, 1, 3])?;

        let k_t = k.transpose(&[0, 1, 3, 2])?;
        let scores = q.matmul(&k_t)?;
        let scores = (&scores / (HEAD_DIM as f64).sqrt())?;

        let attn = scores.softmax(-1)?;

        let out = attn.matmul(&v)?;

        let out = out
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[batch, seq, N_HEADS * HEAD_DIM])?;

        self.o_proj.forward(&out)
    }
}

// ── MLP (SiLU-gated) ───────────────────────────────────────────────

#[derive(Module)]
pub struct Mlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Mlp {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let gate = self.gate_proj.forward(x)?.silu();
        let up = self.up_proj.forward(x)?;
        let hidden = (&gate * &up)?;
        self.down_proj.forward(&hidden)
    }
}

// ── Transformer Layer ───────────────────────────────────────────────

#[derive(Module)]
pub struct TransformerLayer {
    pub attn_norm: RmsNorm,
    pub attn: Attention,
    pub ffn_norm: RmsNorm,
    pub mlp: Mlp,
}

impl TransformerLayer {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let normed = self.attn_norm.forward(x)?;
        let attn_out = self.attn.forward(&normed)?;
        let x = (x + &attn_out)?;

        let normed = self.ffn_norm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        &x + &mlp_out
    }
}

// ── Main ────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ckpt = Checkpoint::from_file("transformer.safetensors")?;
    let device = Device::cpu();

    let batch = 1;
    let seq = 4;
    let dim = 16;

    let runner = device.compile("main", |cx| {
        let x = cx.input("x", Shape::new(vec![batch, seq, dim], DType::F32));
        let layer = TransformerLayer::trace(cx, "layer", ckpt.shapes())?;
        layer.forward(&x)
    })?;

    let weights = ckpt.load_weights(runner.signature())?;
    let sess = runner.session(weights);

    let input_data: Vec<f32> = vec![1.0; (batch * seq * dim) as usize];

    let y = sess.run(|inputs| inputs.set_input("x", input_data.clone()))?;

    println!("output shape: {:?}", y.shape());
    let data = y.to_f32().unwrap();
    println!("first 8 values: {:?}", &data[..8]);
    println!("output sum: {:.6}", data.iter().sum::<f32>());
    Ok(())
}
