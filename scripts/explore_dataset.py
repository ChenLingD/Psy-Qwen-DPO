"""
explore_dataset.py — 探索 PsyDTCorpus 数据结构

输出:
  - 数据集大小
  - 字段名
  - 单条样本的结构
  - 对话轮数分布
  - 每个 split 是什么样
"""

import json
from collections import Counter
from modelscope.msdatasets import MsDataset

print("=" * 70)
print("Step 1: 加载 PsyDTCorpus 数据集")
print("=" * 70)

# 先看有哪些 split (test/train/val)
print("\n尝试加载 test split...")
ds_test = MsDataset.load(
    "YIRONGCHEN/PsyDTCorpus",
    subset_name="default",
    split="train"
)
print(f"  test split size: {len(ds_test)}")

print("\n" + "=" * 70)
print("Step 2: 检查字段结构")
print("=" * 70)
sample = ds_test[0]
print(f"\n字段名: {list(sample.keys())}")
print(f"\n第一条样本各字段类型:")
for k, v in sample.items():
    if isinstance(v, list):
        print(f"  {k}: list, 长度 = {len(v)}")
        if v and isinstance(v[0], dict):
            print(f"     第一项字段: {list(v[0].keys())}")
    elif isinstance(v, str):
        print(f"  {k}: str, 长度 = {len(v)}, 预览: {v[:60]}...")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")

print("\n" + "=" * 70)
print("Step 3: 看一条完整对话（前 6 轮）")
print("=" * 70)

# 自动检测 messages 字段
msg_field = None
for k in ["messages", "conversation", "dialog", "dialogue"]:
    if k in sample:
        msg_field = k
        break
print(f"\n对话字段名: {msg_field}\n")

for i, msg in enumerate(sample[msg_field][:6]):
    role = msg.get("role", "?")
    content = msg.get("content", "")
    print(f"[{i+1}] {role}:")
    print(f"    {content[:200]}{'...' if len(content) > 200 else ''}\n")

print("=" * 70)
print("Step 4: 统计对话轮数分布")
print("=" * 70)
turn_counts = [len(item[msg_field]) for item in ds_test]
turn_dist = Counter()
for t in turn_counts:
    if t < 10:
        turn_dist["< 10"] += 1
    elif t < 20:
        turn_dist["10-19"] += 1
    elif t < 30:
        turn_dist["20-29"] += 1
    elif t < 50:
        turn_dist["30-49"] += 1
    else:
        turn_dist[">= 50"] += 1

print(f"\n总对话数: {len(ds_test)}")
print(f"对话轮数: 最少 {min(turn_counts)}, 最多 {max(turn_counts)}, 平均 {sum(turn_counts)/len(turn_counts):.1f}")
print(f"\n分布:")
for bucket in ["< 10", "10-19", "20-29", "30-49", ">= 50"]:
    count = turn_dist.get(bucket, 0)
    bar = "█" * int(count / 5)
    print(f"  {bucket:>8}: {count:>4} {bar}")

print("\n" + "=" * 70)
print("Step 5: 统计 role 类型")
print("=" * 70)
role_counter = Counter()
for item in ds_test:
    for msg in item[msg_field]:
        role_counter[msg.get("role", "unknown")] += 1
print(f"\n所有 role 类型: {dict(role_counter)}")

print("\n" + "=" * 70)
print("Step 6: 估算可用 prompt 数量")
print("=" * 70)
# 假设每个对话取 2 个截断点（不算第一个 assistant 轮）
total_prompts = 0
for item in ds_test:
    asst_indices = [i for i, m in enumerate(item[msg_field]) if m["role"] == "assistant"]
    if len(asst_indices) > 1:
        total_prompts += min(2, len(asst_indices) - 1)

print(f"\n如果每个对话取 2 个截断点（跳过第一轮 assistant）:")
print(f"  总可用 prompts: {total_prompts}")
print(f"  采 800 个的话覆盖率: {min(800, total_prompts) / total_prompts * 100:.1f}%")

print("\n✅ 数据探索完成")
