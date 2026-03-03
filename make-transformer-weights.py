# /// script
# dependencies = [
#   "numpy",
#   "safetensors",
# ]
# ///

import numpy as np
from safetensors.numpy import save_file

rng = np.random.default_rng(123)

# Single-layer transformer: dim=16, heads=2, head_dim=8, mlp_hidden=32
dim = 16
n_heads = 2
head_dim = dim // n_heads  # 8
mlp_hidden = 32

def init(shape):
    return (rng.standard_normal(shape).astype(np.float32) * 0.02)

tensors = {
    # RMSNorm (pre-attention)
    "layer/attn_norm/weight":       np.ones(dim, dtype=np.float32),

    # Attention projections
    "layer/attn/q_proj/w":          init((dim, dim)),
    "layer/attn/k_proj/w":          init((dim, dim)),
    "layer/attn/v_proj/w":          init((dim, dim)),
    "layer/attn/o_proj/w":          init((dim, dim)),

    # RMSNorm (pre-MLP)
    "layer/ffn_norm/weight":        np.ones(dim, dtype=np.float32),

    # MLP: gate + up project to mlp_hidden, then down back to dim
    "layer/mlp/gate_proj/w":        init((dim, mlp_hidden)),
    "layer/mlp/up_proj/w":          init((dim, mlp_hidden)),
    "layer/mlp/down_proj/w":        init((mlp_hidden, dim)),
}

save_file(tensors, "transformer.safetensors")
print("saved transformer.safetensors with shapes:")
for k, v in tensors.items():
    print(f"  {k}: {v.shape}  {v.dtype}")
