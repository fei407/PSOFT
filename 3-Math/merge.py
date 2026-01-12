import os
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# === 1) Configure checkpoint paths ===
ckpt_dir = "/root/workspace/PSOFT/Math/results/metamath40k/Llama-3.2-3B/H100_qGOFT_peft_lr-5e-4-qkv"
out_dir = ckpt_dir.rstrip("/") + "-merged"

os.makedirs(out_dir, exist_ok=True)

# === 2) Load tokenizer ===
# Try loading directly from the PEFT checkpoint directory;
# if it fails, fall back to loading from the base model path.
try:
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
except Exception:
    # Some PEFT checkpoints store `base_model_name_or_path`,
    # which will be used later by AutoPeftModel.
    tokenizer = None

# === 3) Load the PEFT model ===
# NOTE: Do NOT use device_map="auto" or offloading here,
# to avoid layers being split across different devices.
model = AutoPeftModelForCausalLM.from_pretrained(
    ckpt_dir,
    torch_dtype="auto",   # Can be set to torch.float16 / torch.bfloat16 / torch.float32 if needed
    device_map=None
)

# If the tokenizer was not successfully loaded, try loading it
# from the base model path as a fallback.
if tokenizer is None:
    base = getattr(model, "base_model", None)
    base_name = getattr(model, "base_model_name_or_path", None)
    if base_name is None and base is not None:
        base_name = getattr(base, "name_or_path", None)
    if base_name:
        tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
    else:
        # Final fallback: try loading again from the checkpoint directory
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)

# === 4) Merge adapters following the same logic as discussed ===
# Option A: Move everything to CPU before merging
# (most robust approach, avoids CPU/CUDA tensor mixing)
print(">> Moving model to CPU for safe merge…")
model.to("cpu")
torch.cuda.empty_cache()

print(">> Merging adapters into base weights (merge_and_unload)…")
with torch.no_grad():
    model = model.merge_and_unload()

# (Optional) Ensure parameters are stored contiguously
# for more efficient saving/loading later.
for p in model.parameters():
    if not p.is_contiguous():
        p.data = p.data.contiguous()

# === 5) Save the merged full model ===
print(f">> Saving merged model to: {out_dir}")
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
print(">> Done.")

# === 6) (Optional) Quick sanity check ===
# Perform a minimal text generation on CPU.
# from transformers import pipeline, TextStreamer
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map=None)
# print(pipe("Hello, my name is", max_new_tokens=20)[0]["generated_text"])
