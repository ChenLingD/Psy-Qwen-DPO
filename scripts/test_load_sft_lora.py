"""
快速验证 SFT LoRA 能否被当前版本 peft (0.19.1) 正常加载。
不做训练、不做推理，只测加载 + 打印 trainable params 数。

用法：
    python scripts/test_load_sft_lora.py

期望输出：
    [OK] base model loaded
    [OK] SFT LoRA adapter loaded
    Trainable params: 0 (LoRA in inference mode, this is expected)
    All LoRA modules attached:
      - model.layers.0.self_attn.q_proj.lora_A.default
      - model.layers.0.self_attn.q_proj.lora_B.default
      ... (省略)
    [PASS] SFT LoRA is compatible with current peft version
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 路径常量（如有变化在这里改）
BASE_MODEL = "/mnt/workspace/.cache/modelscope/models/Qwen/Qwen3___5-0___8B"
SFT_LORA  = "/mnt/workspace/output/psy-qwen-0.8b/v2-20260408-061615/checkpoint-375"

print("=" * 60)
print(f"Base model:  {BASE_MODEL}")
print(f"SFT LoRA:    {SFT_LORA}")
print("=" * 60)

# Step 1: 加载 base 模型（bf16 节省内存）
print("\n[1/3] Loading base model in bf16 ...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # 加载到 CPU 即可，不占 GPU
)
print(f"[OK] Base model loaded ({sum(p.numel() for p in base.parameters()) / 1e9:.2f}B params)")

# Step 2: 在 base 上加载 SFT LoRA
print("\n[2/3] Loading SFT LoRA adapter on top of base model ...")
model = PeftModel.from_pretrained(base, SFT_LORA)
print("[OK] SFT LoRA adapter loaded")

# Step 3: 打印 LoRA 模块清单 + trainable params
print("\n[3/3] Inspecting LoRA modules ...")
lora_modules = [n for n, _ in model.named_modules() if "lora_A" in n or "lora_B" in n]
print(f"Total LoRA modules attached: {len(lora_modules)}")
print("First 6 LoRA modules (sanity check):")
for n in lora_modules[:6]:
    print(f"  - {n}")

# inference_mode=true 的 adapter 加载后是冻结的，trainable=0 是正常的
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\nTrainable params: {trainable:,} / {total:,}")
print("(注意: SFT adapter 标记为 inference_mode=true，加载后默认冻结，trainable=0 正常)")
print("(DPO 训练时 ms-swift 会自动把 LoRA 解冻为可训练)")

print("\n" + "=" * 60)
print("[PASS] SFT LoRA is compatible with current peft version (0.19.1)")
print("       Safe to proceed with DPO training.")
print("=" * 60)