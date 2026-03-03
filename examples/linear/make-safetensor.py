# /// script
# dependencies = [
#   "numpy",
#   "safetensors",
# ]
# ///

from pathlib import Path
import numpy as np
from safetensors.numpy import save_file

rng = np.random.default_rng(42)

tensors = {
    "proj/gate_proj/weight": rng.standard_normal((4, 8)).astype(np.float32) * 0.1,
    "proj/up/weight":        rng.standard_normal((4, 8)).astype(np.float32) * 0.1,
    "proj/down/weight":      rng.standard_normal((8, 4)).astype(np.float32) * 0.1,
}

out = Path("examples/linear/artifacts")
out.mkdir(parents=True, exist_ok=True)

path = out / "model.safetensors"
save_file(tensors, str(path))
print(f"saved {path} with shapes:")
for k, v in tensors.items():
    print(f"  {k}: {v.shape}")
