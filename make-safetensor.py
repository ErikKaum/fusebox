# /// script
# dependencies = [
#   "numpy",
#   "safetensors",
# ]
# ///

import numpy as np
from safetensors.numpy import save_file

rng = np.random.default_rng(42)

tensors = {
    "proj/gate_proj/weight": rng.standard_normal((4, 8)).astype(np.float32) * 0.1,
    "proj/up/weight":        rng.standard_normal((4, 8)).astype(np.float32) * 0.1,
    "proj/down/weight":      rng.standard_normal((8, 4)).astype(np.float32) * 0.1,
}

save_file(tensors, "model.safetensors")
print("saved model.safetensors with shapes:")
for k, v in tensors.items():
    print(f"  {k}: {v.shape}")
