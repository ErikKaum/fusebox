# /// script
# dependencies = [
#   "huggingface_hub",
# ]
# ///

"""Download SmolLM2-135M weights for fusebox.

No preprocessing needed -- fusebox handles:
  - Key name normalization (dots → slashes)
  - BF16 → F32 conversion at load time
  - Tied lm_head weights (reuses embed_tokens in the model code)
"""

from huggingface_hub import hf_hub_download
import shutil

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
OUT_PATH = "smollm2-135m.safetensors"

print(f"downloading {MODEL_ID} ...")
path = hf_hub_download(MODEL_ID, "model.safetensors")
print(f"  -> {path}")

shutil.copy(path, OUT_PATH)
print(f"saved {OUT_PATH}")
