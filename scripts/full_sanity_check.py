"""
A 方案完整 Sanity Check：
从 782 held-out prompt 中分层抽 5 个（覆盖 5 个 tag），
对比 SFT vs DPO 模型回复，输出人类可读文件供人眼判断。

判断通过条件：≥ 3/5 DPO 更好 → 进 Phase 3
"""

import json
import random
import torch
from pathlib import Path
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============ 路径配置 ============
BASE_MODEL = "/mnt/workspace/.cache/modelscope/models/Qwen/Qwen3___5-0___8B"
SFT_LORA = "/mnt/workspace/output/psy-qwen-0.8b/v2-20260408-061615/checkpoint-375"
DPO_LORA = "/mnt/workspace/output/psy-qwen-dpo-beta01/v1-20260428-033426/checkpoint-162"
PROJECT_DIR = Path("/mnt/workspace/psy-qwen-dpo")
OUTPUT_FILE = PROJECT_DIR / "data" / "sanity_check_5.txt"

# ============ 推理参数（与交接文档一致） ============
GEN_CONFIG = dict(
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.05,
    do_sample=True,
)
SEED = 42

# ============ 抽样配置 ============
# 选 5 个 tag 覆盖不同心理咨询场景（held-out 数量充足的）
TARGET_TAGS = ["婚恋", "职场", "家庭", "情绪", "治疗"]
SAMPLES_PER_TAG = 1


# ============ 工具函数 ============
def get_fp_from_dpo(row):
    """从 dpo_train/val 行里提取最后一条 user content 作指纹"""
    for msg in reversed(row["messages"]):
        if msg["role"] == "user":
            return msg["content"]
    return None


def get_fp_from_prompts(row):
    """从 prompts.jsonl 行里提取最后一条 user content 作指纹"""
    for msg in reversed(row["context"]):
        if msg["role"] == "user":
            return msg["content"]
    return None


def load_held_out_prompts():
    """复用 check_held_out 的逻辑，返回 held-out prompt 列表"""
    train_fps = set()
    with open(PROJECT_DIR / "data" / "dpo_train.jsonl") as f:
        for line in f:
            fp = get_fp_from_dpo(json.loads(line))
            if fp:
                train_fps.add(fp)

    val_fps = set()
    with open(PROJECT_DIR / "data" / "dpo_val.jsonl") as f:
        for line in f:
            fp = get_fp_from_dpo(json.loads(line))
            if fp:
                val_fps.add(fp)

    seen = train_fps | val_fps

    held_out = []
    with open(PROJECT_DIR / "data" / "prompts.jsonl") as f:
        for line in f:
            row = json.loads(line)
            fp = get_fp_from_prompts(row)
            if fp and fp not in seen:
                held_out.append(row)

    print(f"[Data] train fingerprints: {len(train_fps)}")
    print(f"[Data] val fingerprints:   {len(val_fps)}")
    print(f"[Data] held-out total:     {len(held_out)}")
    return held_out


def stratified_sample(held_out, target_tags, samples_per_tag, seed=SEED):
    """分层抽样：每个 target tag 抽 samples_per_tag 个"""
    random.seed(seed)
    by_tag = defaultdict(list)
    for row in held_out:
        if row["tag"] in target_tags:
            by_tag[row["tag"]].append(row)

    picked = []
    for tag in target_tags:
        candidates = by_tag.get(tag, [])
        if len(candidates) < samples_per_tag:
            print(f"⚠️  tag '{tag}' only has {len(candidates)} held-out, "
                  f"need {samples_per_tag}")
        picked.extend(random.sample(candidates, min(samples_per_tag, len(candidates))))
    return picked


def generate_reply(model, tokenizer, messages):
    """单次生成：messages = list of {role, content}"""
    # apply chat template，加 generation prompt（assistant: 前缀）
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    # 重置 seed 保证 SFT vs DPO 生成的随机性一致
    torch.manual_seed(SEED)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **GEN_CONFIG,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # 只取新生成部分
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return reply


def format_context_preview(context, max_turns=3):
    """格式化对话历史预览：跳过 system，只展示最后 max_turns 轮"""
    non_system = [m for m in context if m["role"] != "system"]
    # 最多保留 max_turns*2 条消息（一轮 = user + assistant）
    if len(non_system) > max_turns * 2:
        preview = non_system[-(max_turns * 2):]
        prefix = f"...（省略前 {len(non_system) - max_turns * 2} 条对话）...\n\n"
    else:
        preview = non_system
        prefix = ""

    lines = [prefix] if prefix else []
    for m in preview:
        role_zh = "来访者" if m["role"] == "user" else "咨询师"
        lines.append(f"【{role_zh}】{m['content']}")
    return "\n".join(lines)


# ============ 主流程 ============
def main():
    # ----- 1. 抽样 -----
    print("=" * 60)
    print("Step 1/4: 加载 held-out 并抽样")
    print("=" * 60)
    held_out = load_held_out_prompts()
    samples = stratified_sample(held_out, TARGET_TAGS, SAMPLES_PER_TAG)
    print(f"[Sample] picked {len(samples)} prompts:")
    for s in samples:
        print(f"  - [{s['tag']}] {s['prompt_id']} (context len={len(s['context'])})")

    # ----- 2. 加载基座 + tokenizer -----
    print("\n" + "=" * 60)
    print("Step 2/4: 加载基座模型 + tokenizer")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()
    print(f"[Model] base loaded, device={base_model.device}")

    # ----- 3. SFT 推理 -----
    print("\n" + "=" * 60)
    print("Step 3/4: 加载 SFT LoRA 并生成 5 个回复")
    print("=" * 60)
    sft_model = PeftModel.from_pretrained(base_model, SFT_LORA, adapter_name="sft")
    sft_model.eval()
    sft_replies = []
    for i, s in enumerate(samples, 1):
        print(f"  [SFT {i}/{len(samples)}] {s['tag']} ...", end=" ", flush=True)
        reply = generate_reply(sft_model, tokenizer, s["context"])
        sft_replies.append(reply)
        print(f"({len(reply)} 字符)")

    # 卸载 SFT adapter，回到纯基座
    sft_model = sft_model.unload()
    del sft_model
    torch.cuda.empty_cache()
    print("[Model] SFT adapter unloaded")

    # ----- 4. DPO 推理 -----
    print("\n" + "=" * 60)
    print("Step 4/4: 加载 DPO LoRA 并生成 5 个回复")
    print("=" * 60)
    dpo_model = PeftModel.from_pretrained(base_model, DPO_LORA, adapter_name="dpo")
    dpo_model.eval()
    dpo_replies = []
    for i, s in enumerate(samples, 1):
        print(f"  [DPO {i}/{len(samples)}] {s['tag']} ...", end=" ", flush=True)
        reply = generate_reply(dpo_model, tokenizer, s["context"])
        dpo_replies.append(reply)
        print(f"({len(reply)} 字符)")

    # ----- 5. 写文件 -----
    print("\n" + "=" * 60)
    print("Writing output to:", OUTPUT_FILE)
    print("=" * 60)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Psy-Qwen-DPO Sanity Check (A 方案完整版)\n")
        f.write(f"对比模型：SFT (checkpoint-375) vs DPO (checkpoint-162, β=0.1)\n")
        f.write(f"采样：{len(samples)} prompts，分层覆盖 tags = {TARGET_TAGS}\n")
        f.write(f"推理参数：T=0.7, top_p=0.9, rep_pen=1.05, seed={SEED}\n")
        f.write("=" * 70 + "\n\n")

        for i, (s, sft, dpo) in enumerate(zip(samples, sft_replies, dpo_replies), 1):
            f.write(f"\n{'#' * 70}\n")
            f.write(f"### 案例 {i} / {len(samples)}  |  tag = {s['tag']}  |  "
                    f"prompt_id = {s['prompt_id']}\n")
            f.write(f"{'#' * 70}\n\n")

            f.write("【对话历史】（已截断）\n")
            f.write("-" * 70 + "\n")
            f.write(format_context_preview(s["context"], max_turns=3))
            f.write("\n\n")

            f.write("【SFT 回复】(" + str(len(sft)) + " 字符)\n")
            f.write("-" * 70 + "\n")
            f.write(sft + "\n\n")

            f.write("【DPO 回复】(" + str(len(dpo)) + " 字符)\n")
            f.write("-" * 70 + "\n")
            f.write(dpo + "\n\n")

            f.write("【参考：原 ground_truth】\n")
            f.write("-" * 70 + "\n")
            f.write(s.get("ground_truth", "（无）") + "\n")

        f.write("\n\n" + "=" * 70 + "\n")
        f.write("人眼判断格式（贴回 Claude）：\n")
        f.write("案例 1 [tag]: SFT 更好 / DPO 更好 / 平 — 理由\n")
        f.write("案例 2 [tag]: ...\n")
        f.write("...\n")
        f.write("=" * 70 + "\n")

    print(f"\n✅ Done! 请打开 {OUTPUT_FILE} 查看")
    print(f"   通过条件：≥ 3/5 DPO 更好 → 进 Phase 3")


if __name__ == "__main__":
    main()