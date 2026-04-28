"""
Phase 3a: 为 200 个 held-out prompt 生成 SFT + DPO 回复。

输出：data/phase3_generations.jsonl
每行：{prompt_id, tag, context, sft_reply, dpo_reply}

支持断点续跑：启动时读已写入的 prompt_id，跳过已完成的。
"""

import json
import random
import torch
from pathlib import Path
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============ 配置 ============
BASE_MODEL = "/mnt/workspace/.cache/modelscope/models/Qwen/Qwen3___5-0___8B"
SFT_LORA = "/mnt/workspace/output/psy-qwen-0.8b/v2-20260408-061615/checkpoint-375"
DPO_LORA = "/mnt/workspace/output/psy-qwen-dpo-beta01/v1-20260428-033426/checkpoint-162"
PROJECT_DIR = Path("/mnt/workspace/psy-qwen-dpo")
OUTPUT_FILE = PROJECT_DIR / "data" / "phase3_generations.jsonl"

TARGET_N = 200
SEED = 42

GEN_CONFIG = dict(
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.05,
    do_sample=True,
)


# ============ 工具函数 ============
def get_fp_from_dpo(row):
    for msg in reversed(row["messages"]):
        if msg["role"] == "user":
            return msg["content"]
    return None


def get_fp_from_prompts(row):
    for msg in reversed(row["context"]):
        if msg["role"] == "user":
            return msg["content"]
    return None


def load_held_out():
    train_fps, val_fps = set(), set()
    with open(PROJECT_DIR / "data" / "dpo_train.jsonl") as f:
        for line in f:
            fp = get_fp_from_dpo(json.loads(line))
            if fp:
                train_fps.add(fp)
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
    return held_out


def stratified_sample(held_out, target_n, seed=SEED):
    """按 tag 在 held-out 里的比例分层抽 target_n 个"""
    random.seed(seed)
    by_tag = defaultdict(list)
    for row in held_out:
        by_tag[row["tag"]].append(row)

    total = len(held_out)
    picked = []
    plan = {}
    for tag, rows in by_tag.items():
        k = round(len(rows) / total * target_n)
        plan[tag] = (len(rows), k)
        picked.extend(random.sample(rows, min(k, len(rows))))

    print(f"[Sample] target={target_n}, actual={len(picked)}")
    print(f"[Sample] plan (held-out -> picked):")
    for tag, (avail, k) in sorted(plan.items(), key=lambda x: -x[1][0]):
        print(f"  {tag}: {avail} -> {k}")
    return picked


def load_done_prompt_ids():
    if not OUTPUT_FILE.exists():
        return set()
    done = set()
    with open(OUTPUT_FILE) as f:
        for line in f:
            try:
                done.add(json.loads(line)["prompt_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def generate_reply(model, tokenizer, messages):
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    torch.manual_seed(SEED)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **GEN_CONFIG,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============ 主流程 ============
def main():
    import time
    t_start = time.time()

    print("=" * 60)
    print("Phase 3a: 生成 SFT + DPO 回复")
    print("=" * 60)

    # 1. 抽样
    held_out = load_held_out()
    print(f"[Data] held-out total: {len(held_out)}")
    samples = stratified_sample(held_out, TARGET_N)

    # 2. 断点续跑：跳过已完成
    done = load_done_prompt_ids()
    if done:
        print(f"\n[Resume] {len(done)} prompts already done, skipping them")
        samples = [s for s in samples if s["prompt_id"] not in done]
    print(f"[Resume] {len(samples)} prompts to process")

    if not samples:
        print("✅ 所有样本已完成，无需重跑")
        return

    # 3. 先把 SFT 全跑完
    print("\n" + "=" * 60)
    print("Stage 1: 加载基座 + tokenizer")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    print("\n" + "=" * 60)
    print(f"Stage 2: SFT 生成 ({len(samples)} 个)")
    print("=" * 60)
    sft_model = PeftModel.from_pretrained(base_model, SFT_LORA, adapter_name="sft")
    sft_model.eval()
    sft_replies = {}
    for i, s in enumerate(samples, 1):
        reply = generate_reply(sft_model, tokenizer, s["context"])
        sft_replies[s["prompt_id"]] = reply
        if i % 10 == 0 or i == len(samples):
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(samples) - i)
            print(f"  [SFT {i}/{len(samples)}] elapsed={elapsed:.0f}s, eta={eta:.0f}s")

    sft_model = sft_model.unload()
    del sft_model
    torch.cuda.empty_cache()
    print("[Model] SFT unloaded")

    # 4. DPO + 增量写文件
    print("\n" + "=" * 60)
    print(f"Stage 3: DPO 生成 + 增量写文件 ({len(samples)} 个)")
    print("=" * 60)
    dpo_model = PeftModel.from_pretrained(base_model, DPO_LORA, adapter_name="dpo")
    dpo_model.eval()

    # 用 'a' 模式追加，断点续跑友好
    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        t_dpo_start = time.time()
        for i, s in enumerate(samples, 1):
            dpo_reply = generate_reply(dpo_model, tokenizer, s["context"])
            record = {
                "prompt_id": s["prompt_id"],
                "tag": s["tag"],
                "context": s["context"],
                "sft_reply": sft_replies[s["prompt_id"]],
                "dpo_reply": dpo_reply,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()  # 立即写入，断点续跑安全

            if i % 10 == 0 or i == len(samples):
                elapsed = time.time() - t_dpo_start
                eta = elapsed / i * (len(samples) - i)
                print(f"  [DPO {i}/{len(samples)}] elapsed={elapsed:.0f}s, eta={eta:.0f}s")

    total = time.time() - t_start
    print(f"\n✅ Done! Total time: {total:.0f}s ({total/60:.1f} min)")
    print(f"   Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()