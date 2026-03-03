# /// script
# dependencies = [
#   "huggingface_hub",
# ]
# ///

"""Download SmolLM2-135M weights and tokenizer for fusebox.

No preprocessing needed -- fusebox handles:
  - Key name normalization (dots → slashes)
  - BF16 → F32 conversion at load time
  - Tied lm_head weights (reuses embed_tokens in the model code)
"""

from pathlib import Path

from huggingface_hub import hf_hub_download
import shutil

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

WEIGHTS_OUT = Path("examples/smollm2/artifacts/smollm2-135m.safetensors")
TOKENIZER_OUT = Path("examples/smollm2/artifacts/tokenizer.json")

WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)

print(f"downloading {MODEL_ID} weights ...")
weights_path = hf_hub_download(MODEL_ID, "model.safetensors")
shutil.copy(weights_path, WEIGHTS_OUT)
print(f"  saved {WEIGHTS_OUT}")

print(f"downloading {MODEL_ID} tokenizer ...")
tokenizer_path = hf_hub_download(MODEL_ID, "tokenizer.json")
shutil.copy(tokenizer_path, TOKENIZER_OUT)
print(f"  saved {TOKENIZER_OUT}")
