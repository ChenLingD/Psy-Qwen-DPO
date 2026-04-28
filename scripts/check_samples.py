"""Sanity check for raw_samples.jsonl before scoring."""
import json
import random
from collections import Counter
from pathlib import Path

PATH = Path("/mnt/workspace/psy-qwen-dpo/data/raw_samples.jsonl")

records = []
with open(PATH, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"总记录数: {len(records)}")
print(f"第一条的 keys: {list(records[0].keys())}")
print()

sample_key = None
for k in ["samples", "responses", "candidates", "completions"]:
    if k in records[0]:
        sample_key = k
        break
print(f"Samples 字段名: {sample_key}")
print()

n_samples_per_prompt = Counter(len(r[sample_key]) for r in records)
print(f"[1] 每 prompt 的 sample 数分布: {dict(n_samples_per_prompt)}")

all_samples = [s for r in records for s in r[sample_key]]
empty = sum(1 for s in all_samples if not s or not s.strip())
print(f"[2] 总 sample 数: {len(all_samples)}, 空回复: {empty} ({empty/len(all_samples)*100:.2f}%)")

diversity = [len(set(r[sample_key])) for r in records]
div_counter = Counter(diversity)
print(f"[3] 同 prompt 内去重后 unique 数分布: {dict(sorted(div_counter.items()))}")
print(f"    平均 unique 数: {sum(diversity)/len(diversity):.2f}")

lens = [len(s) for s in all_samples if s]
lens.sort()
n = len(lens)
print(f"[4] 回复长度（字符数）:")
print(f"    min={lens[0]}, p25={lens[n//4]}, median={lens[n//2]}, p75={lens[3*n//4]}, max={lens[-1]}")
print(f"    平均: {sum(lens)/n:.1f}")
near_max = sum(1 for l in lens if l >= 290)
print(f"    >= 290 字符（疑似被截断）: {near_max} ({near_max/n*100:.2f}%)")

print()
print("=" * 60)
print("[5] 随机抽看 3 条样本:")
print("=" * 60)
random.seed(42)
for r in random.sample(records, 3):
    print(f"\n--- ID: {r.get('id', 'N/A')} | topic: {r.get('topic', r.get('normalizedTag', 'N/A'))} ---")
    if "messages" in r:
        last_user = [m for m in r["messages"] if m["role"] == "user"][-1]["content"]
        print(f"最后一句 user: {last_user[:100]}...")
    elif "prompt" in r:
        print(f"Prompt 末尾: ...{r['prompt'][-150:]}")
    for i, s in enumerate(r[sample_key]):
        preview = s[:120] + ("..." if len(s) > 120 else "")
        print(f"  [Sample {i+1}] ({len(s)} 字符) {preview}")
