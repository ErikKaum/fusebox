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

# Weights use [out_features, in_features] layout (PyTorch / HuggingFace convention).
tensors = {
    # RMSNorm (pre-attention)
    "layer/attn_norm/weight":       np.ones(dim, dtype=np.float32),

    # Attention projections: all [dim, dim] = [out, in]
    "layer/attn/q_proj/weight":     init((dim, dim)),
    "layer/attn/k_proj/weight":     init((dim, dim)),
    "layer/attn/v_proj/weight":     init((dim, dim)),
    "layer/attn/o_proj/weight":     init((dim, dim)),

    # RMSNorm (pre-MLP)
    "layer/ffn_norm/weight":        np.ones(dim, dtype=np.float32),

    # MLP: gate + up project to mlp_hidden, then down back to dim
    "layer/mlp/gate_proj/weight":   init((mlp_hidden, dim)),
    "layer/mlp/up_proj/weight":     init((mlp_hidden, dim)),
    "layer/mlp/down_proj/weight":   init((dim, mlp_hidden)),
}

save_file(tensors, "transformer.safetensors")
print("saved transformer.safetensors with shapes:")
for k, v in tensors.items():
    print(f"  {k}: {v.shape}  {v.dtype}")
