"""
B 方案 Sanity Check：1 个 prompt 测试 DPO 模型是否输出正常 + 对比 SFT。
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---- 路径 ----
BASE_MODEL = "/mnt/workspace/.cache/modelscope/models/Qwen/Qwen3___5-0___8B"
SFT_LORA   = "/mnt/workspace/output/psy-qwen-0.8b/v2-20260408-061615/checkpoint-375"
DPO_LORA   = "/mnt/workspace/output/psy-qwen-dpo-beta01/v1-20260428-033426/checkpoint-162"

# ---- 取 PsyDTCorpus 真实 system prompt（738 字符）----
with open("/mnt/workspace/psy-qwen-dpo/data/dpo_train.jsonl") as f:
    first_row = json.loads(f.readline())
    SYSTEM_PROMPT = first_row["messages"][0]["content"] if first_row["messages"][0]["role"] == "system" else None

if SYSTEM_PROMPT is None:
    raise ValueError("Could not extract system prompt from dpo_train.jsonl row 1")

print(f"System prompt: {SYSTEM_PROMPT[:100]}... (total {len(SYSTEM_PROMPT)} chars)")
print()

# ---- 测试 prompt（典型心理咨询场景）----
TEST_USER_MSG = "我最近压力很大，工作上感觉做什么都不对，老板总是挑刺，我开始怀疑自己是不是不适合这份工作了。"

# ---- 推理工具函数 ----
def generate(model, tokenizer, system_prompt, user_msg, max_new=200):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ---- Step 1: SFT ----
print("=" * 70)
print("[1/3] Loading SFT LoRA on base model ...")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_sft = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0",
)
sft_model = PeftModel.from_pretrained(base_sft, SFT_LORA)
sft_model.eval()
print("[OK] SFT model loaded")

print("\nGenerating SFT response (~30s) ...")
torch.manual_seed(42)
sft_response = generate(sft_model, tokenizer, SYSTEM_PROMPT, TEST_USER_MSG)
print("[OK] SFT response generated")

del sft_model
del base_sft
torch.cuda.empty_cache()

# ---- Step 2: DPO ----
print("\n" + "=" * 70)
print("[2/3] Loading DPO LoRA on (fresh) base model ...")
print("=" * 70)

base_dpo = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0",
)
dpo_model = PeftModel.from_pretrained(base_dpo, DPO_LORA)
dpo_model.eval()
print("[OK] DPO model loaded")

print("\nGenerating DPO response (~30s) ...")
torch.manual_seed(42)
dpo_response = generate(dpo_model, tokenizer, SYSTEM_PROMPT, TEST_USER_MSG)
print("[OK] DPO response generated")

# ---- Step 3: 并排打印 ----
print("\n" + "=" * 70)
print("[3/3] SANITY CHECK RESULTS")
print("=" * 70)
print(f"\n用户问题:\n{TEST_USER_MSG}\n")

print("-" * 70)
print(f"📘 SFT 回复 (chars: {len(sft_response)}):")
print("-" * 70)
print(sft_response)
print()

print("-" * 70)
print(f"🆕 DPO 回复 (chars: {len(dpo_response)}):")
print("-" * 70)
print(dpo_response)
print()

# ---- 自动检查 ----
print("=" * 70)
print("AUTO-CHECKS:")
print("=" * 70)

checks = []
sft_len_ok = 10 < len(sft_response) < 500
dpo_len_ok = 10 < len(dpo_response) < 500
checks.append(("SFT length reasonable (10-500)", sft_len_ok, f"{len(sft_response)} chars"))
checks.append(("DPO length reasonable (10-500)", dpo_len_ok, f"{len(dpo_response)} chars"))

def has_loop(text, window=4, min_repeats=4):
    if len(text) < 30:
        return False
    tail = text[-50:]
    for i in range(len(tail) - window * min_repeats):
        substring = tail[i:i+window]
        if tail[i:i+window*min_repeats] == substring * min_repeats:
            return True
    return False

checks.append(("SFT no token loop", not has_loop(sft_response), ""))
checks.append(("DPO no token loop", not has_loop(dpo_response), ""))

sft_has_zh = any('\u4e00' <= c <= '\u9fff' for c in sft_response)
dpo_has_zh = any('\u4e00' <= c <= '\u9fff' for c in dpo_response)
checks.append(("SFT contains Chinese chars", sft_has_zh, ""))
checks.append(("DPO contains Chinese chars", dpo_has_zh, ""))

print()
all_pass = True
for name, ok, info in checks:
    status = "✅" if ok else "❌"
    extra = f" ({info})" if info else ""
    print(f"  {status} {name}{extra}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print("=" * 70)
    print("[PASS] DPO output looks healthy. Ready for full Step 2.4 sanity check.")
    print("=" * 70)
else:
    print("=" * 70)
    print("[WARN] Some auto-checks failed. Please inspect outputs above carefully.")
    print("=" * 70)