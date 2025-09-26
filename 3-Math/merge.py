import os
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# === 1) 配置你的 checkpoint 路径 ===
ckpt_dir = "/root/workspace/MOFT_ICLR/Math/results/metamath40k/Llama-3.2-3B/H100_qGOFT_peft_lr-5e-4-qkv"
out_dir = ckpt_dir.rstrip("/")+ "-merged"

os.makedirs(out_dir, exist_ok=True)

# === 2) 加载 tokenizer（直接从 peft 目录读；若失败再从 base 模型名回退读取） ===
try:
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
except Exception:
    # 有些 peft 目录带有 base_model_name_or_path 信息；AutoPeftModel 会用到它
    tokenizer = None

# === 3) 加载 PEFT 模型（AutoPeft 会自动找到 base 并装上 adapters） ===
# 关键：不使用 device_map="auto"/offload，避免层分散到不同设备
model = AutoPeftModelForCausalLM.from_pretrained(
    ckpt_dir,
    torch_dtype="auto",   # 按需也可改成 torch.float16 / torch.bfloat16 / torch.float32
    device_map=None
)

# 如果 tokenizer 还没加载成功，尽量从模型的 base 路径再加载一次
if tokenizer is None:
    base = getattr(model, "base_model", None)
    base_name = getattr(model, "base_model_name_or_path", None)
    if base_name is None and base is not None:
        base_name = getattr(base, "name_or_path", None)
    if base_name:
        tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
    else:
        # 兜底：再尝试从 ckpt_dir 读取
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)

# === 4) 按“刚才的逻辑”：统一设备 -> 合并 ===
# 方案A：统一到 CPU 再合并（最稳妥，避免 cpu/cuda 混用）
print(">> Moving model to CPU for safe merge…")
model.to("cpu")
torch.cuda.empty_cache()

print(">> Merging adapters into base weights (merge_and_unload)…")
with torch.no_grad():
    model = model.merge_and_unload()

# （可选）把参数排布弄成连续，便于后续保存/加载
for p in model.parameters():
    if not p.is_contiguous():
        p.data = p.data.contiguous()

# === 5) 保存合并后的“整模型” ===
print(f">> Saving merged model to: {out_dir}")
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
print(">> Done.")

# === 6) （可选）快速自检：在 CPU 上做一次极简生成
# from transformers import pipeline, TextStreamer
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map=None)
# print(pipe("Hello, my name is", max_new_tokens=20)[0]["generated_text"])
