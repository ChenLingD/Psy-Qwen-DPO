"""
检查 prompts.jsonl 里有多少 prompt 没在 dpo_train + dpo_val 里出现，
以及在 12 个心理咨询 tag 上的分布。
"""
import json
from collections import Counter

def get_fp_from_dpo(row):
    """从 dpo_train/val 行里提取最后一条 user content 作指纹"""
    for msg in reversed(row['messages']):
        if msg['role'] == 'user':
            return msg['content']
    return None

def get_fp_from_prompts(row):
    """从 prompts.jsonl 行里提取最后一条 user content 作指纹"""
    for msg in reversed(row['context']):
        if msg['role'] == 'user':
            return msg['content']
    return None

# Step 1: 收集 train + val fingerprints
train_fps = set()
with open('data/dpo_train.jsonl') as f:
    for line in f:
        fp = get_fp_from_dpo(json.loads(line))
        if fp:
            train_fps.add(fp)

val_fps = set()
with open('data/dpo_val.jsonl') as f:
    for line in f:
        fp = get_fp_from_dpo(json.loads(line))
        if fp:
            val_fps.add(fp)

print(f'Train unique prompt fingerprints: {len(train_fps)}')
print(f'Val unique prompt fingerprints:   {len(val_fps)}')
print(f'Train + Val total unique:         {len(train_fps | val_fps)}')

# Step 2: 从 prompts.jsonl 找 held-out
prompts_fps = []
with open('data/prompts.jsonl') as f:
    for line in f:
        row = json.loads(line)
        fp = get_fp_from_prompts(row)
        prompts_fps.append((row['prompt_id'], row['tag'], fp))

print(f'\nTotal prompts in prompts.jsonl:   {len(prompts_fps)}')

seen = train_fps | val_fps
held_out = [p for p in prompts_fps if p[2] not in seen]
print(f'Held-out (not in train+val):      {len(held_out)}')

# Step 3: 看 held-out 在 tag 上的分布
held_out_tags = Counter(p[1] for p in held_out)
print(f'\nHeld-out tag distribution:')
for tag, n in sorted(held_out_tags.items(), key=lambda x: -x[1]):
    print(f'  {tag}: {n}')