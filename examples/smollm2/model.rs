use fusebox::prelude::*;

// ── SmolLM2-135M config ─────────────────────────────────────────────

const HIDDEN: i64 = 576;
const N_HEADS: i64 = 9;
const N_KV_HEADS: i64 = 3;
const HEAD_DIM: i64 = 64;
const ROPE_THETA: f64 = 10000.0;
const RMS_EPS: f64 = 1e-5;
const N_GROUPS: i64 = N_HEADS / N_KV_HEADS;

// ── RoPE ────────────────────────────────────────────────────────────

fn build_rope(cx: &mut TraceCx, seq: i64) -> Result<(Tensor, Tensor), Error> {
    let positions = cx.iota(Shape::new(vec![seq, 1], DType::I32), 0)?;
    let positions = positions.to_dtype(DType::F32);

    let dim_idx = cx.iota(Shape::new(vec![1, HEAD_DIM / 2], DType::I32), 1)?;
    let dim_idx = dim_idx.to_dtype(DType::F32);

    // theta^(-2i/d) = exp(-2i/d * ln(theta))
    let log_inv_freq = (&dim_idx * (-2.0 * ROPE_THETA.ln() / HEAD_DIM as f64))?;
    let inv_freq = log_inv_freq.exp();

    let angles = positions.matmul(&inv_freq)?;

    Ok((angles.cos(), angles.sin()))
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, seq: i64) -> Result<Tensor, Error> {
    let x1 = x.narrow(-1, 0, HEAD_DIM / 2)?;
    let x2 = x.narrow(-1, HEAD_DIM / 2, HEAD_DIM / 2)?;

    let cos = cos.reshape(&[1, 1, seq, HEAD_DIM / 2])?;
    let sin = sin.reshape(&[1, 1, seq, HEAD_DIM / 2])?;

    let rotated = Tensor::cat(
        &[
            &(&(&x1 * &cos)? - &(&x2 * &sin)?)?,
            &(&(&x2 * &cos)? + &(&x1 * &sin)?)?,
        ],
        -1,
    )?;
    Ok(rotated)
}

fn build_causal_mask(cx: &mut TraceCx, seq: i64) -> Result<Tensor, Error> {
    let row = cx.iota(Shape::new(vec![1, 1, seq, seq], DType::I32), 2)?;
    let col = cx.iota(Shape::new(vec![1, 1, seq, seq], DType::I32), 3)?;
    row.ge(&col)
}

// ── GQA repeat ──────────────────────────────────────────────────────

fn repeat_kv(x: &Tensor, batch: i64, seq: i64) -> Result<Tensor, Error> {
    let x = x.unsqueeze(2)?;
    let x = x.expand(&[batch, N_KV_HEADS, N_GROUPS, seq, HEAD_DIM])?;
    x.reshape(&[batch, N_HEADS, seq, HEAD_DIM])
}

// ── Attention ───────────────────────────────────────────────────────

#[derive(Module)]
pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
}

impl Attention {
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        batch: i64,
        seq: i64,
    ) -> Result<Tensor, Error> {
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape(&[batch, seq, N_HEADS, HEAD_DIM])?
            .transpose(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[batch, seq, N_KV_HEADS, HEAD_DIM])?
            .transpose(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[batch, seq, N_KV_HEADS, HEAD_DIM])?
            .transpose(&[0, 2, 1, 3])?;

        let q = apply_rope(&q, cos, sin, seq)?;
        let k = apply_rope(&k, cos, sin, seq)?;

        let k = repeat_kv(&k, batch, seq)?;
        let v = repeat_kv(&v, batch, seq)?;

        let k_t = k.transpose(&[0, 1, 3, 2])?;
        let scores = q.matmul(&k_t)?;
        let scores = (&scores / (HEAD_DIM as f64).sqrt())?;

        let mask = mask.expand(&[batch, N_HEADS, seq, seq])?;
        let neg_inf = scores.full_like(f64::NEG_INFINITY);
        let scores = Tensor::select(&mask, &scores, &neg_inf)?;

        let attn = scores.softmax(-1)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[batch, seq, HIDDEN])?;
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
        self.down_proj.forward(&gate.mul(&up)?)
    }
}

// ── Transformer Layer ───────────────────────────────────────────────

#[derive(Module)]
pub struct TransformerLayer {
    pub input_layernorm: RmsNorm,
    pub self_attn: Attention,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: Mlp,
}

impl TransformerLayer {
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        batch: i64,
        seq: i64,
    ) -> Result<Tensor, Error> {
        let normed = self.input_layernorm.forward_with_eps(x, RMS_EPS)?;
        let attn_out = self
            .self_attn
            .forward(&normed, cos, sin, mask, batch, seq)?;
        let x = (x + &attn_out)?;

        let normed = self.post_attention_layernorm.forward_with_eps(&x, RMS_EPS)?;
        let mlp_out = self.mlp.forward(&normed)?;
        &x + &mlp_out
    }
}

// ── Top-level model ─────────────────────────────────────────────────

#[derive(Module)]
pub struct SmolLM2Model {
    pub embed_tokens: Embedding,
    pub layers: Vec<TransformerLayer>,
    pub norm: RmsNorm,
}

// ── Trace helper ────────────────────────────────────────────────────

/// Trace the full SmolLM2 graph. Returns the argmax token id at a
/// caller-specified sequence position (shape: [batch], dtype: i32).
///
/// Inputs:
///   - `tokens`:   [batch, seq] i32  — token ids (right-padded with 0)
///   - `last_pos`: [batch]     i32  — index of the last real token per row
pub fn trace_smollm2(
    cx: &mut TraceCx,
    shapes: &dyn ShapeProvider,
    batch: i64,
    seq: i64,
) -> Result<Tensor, Error> {
    let tokens = cx.input("tokens", Shape::new(vec![batch, seq], DType::I32));
    let last_pos = cx.input("last_pos", Shape::new(vec![batch], DType::I32));

    let model = SmolLM2Model::trace(cx, "model", shapes)?;
    let mut h = model.embed_tokens.forward(&tokens)?;

    let (cos, sin) = build_rope(cx, seq)?;
    let mask = build_causal_mask(cx, seq)?;

    for layer in &model.layers {
        h = layer.forward(&h, &cos, &sin, &mask, batch, seq)?;
    }

    h = model.norm.forward_with_eps(&h, RMS_EPS)?;

    // Tied weights: lm_head shares embed_tokens.weight.
    let lm_weight_t = model.embed_tokens.weight.transpose(&[1, 0])?;
    let logits = h.matmul(&lm_weight_t)?;

    // Extract logits at each row's last real token position via one-hot masking.
    let seq_idx = cx.iota(Shape::new(vec![1, seq], DType::I32), 1)?;
    let last_pos_2d = last_pos.unsqueeze(1)?;
    let mask = seq_idx.eq(&last_pos_2d)?;
    let mask_f = mask.to_dtype(DType::F32).unsqueeze(-1)?;
    let gathered = (&logits * &mask_f)?.sum(&[1])?;

    let next_token = gathered.argmax(-1)?;

    Ok(next_token)
}
