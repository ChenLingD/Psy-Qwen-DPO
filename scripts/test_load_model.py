"""
test_load_model.py — 验证 SFT 模型能正确加载并推理

执行: python scripts/test_load_model.py
"""

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from peft import PeftModel

# 配置路径
BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B"
LORA_PATH = "/mnt/workspace/output/psy-qwen-0.8b/v2-20260408-061615/checkpoint-375"

print("=" * 60)
print("Step 1: 下载/加载基座模型")
print("=" * 60)
base_dir = snapshot_download(BASE_MODEL_ID)
print(f"基座模型路径: {base_dir}")

print("\n" + "=" * 60)
print("Step 2: 加载 tokenizer")
print("=" * 60)
tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer 加载完成. Vocab size: {len(tokenizer)}")

print("\n" + "=" * 60)
print("Step 3: 加载基座模型到 GPU (bf16)")
print("=" * 60)
base_model = AutoModelForCausalLM.from_pretrained(
    base_dir,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"基座模型加载完成")
print(f"GPU 显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("\n" + "=" * 60)
print("Step 4: 加载 LoRA adapter")
print("=" * 60)
print(f"LoRA 路径: {LORA_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print(f"LoRA 加载完成")
print(f"GPU 显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("\n" + "=" * 60)
print("Step 5: 测试推理 (心理咨询场景)")
print("=" * 60)

test_messages = [
    {"role": "system", "content": "你是一位专业的心理咨询师，请耐心倾听并给予支持。"},
    {"role": "user", "content": "我最近压力很大，每天都睡不好觉，工作上也总是出错，感觉很累。"},
]

input_text = tokenizer.apply_chat_template(
    test_messages, tokenize=False, add_generation_prompt=True
)
print(f"输入:\n{test_messages[-1]['content']}\n")

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(
    output[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)
print(f"模型回复:\n{response}")

print("\n" + "=" * 60)
print("✅ 全部测试通过！SFT 模型可用。")
print("=" * 60)
