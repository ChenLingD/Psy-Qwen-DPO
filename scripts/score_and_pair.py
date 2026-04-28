"""
Task 1.4 (v2.2): Score 4 candidate responses + construct chosen/rejected pairs.

CHANGES FROM v2 (based on 2nd error analysis):
  1. Loosened length floor: <15 chars -2 (was <20). 15-30 chars now +0.5 (was 0).
     Reason: Chinese "那您当时感受是什么？" (16 chars) is a great open question.
  2. good_question now exempts empathy_first penalty:
     Pure-question responses don't need an empathy opener.
  3. Per-tag MIN_GAP: 心理学知识 uses 3.5 (others 4.0).
     Reason: pure-knowledge tag has fewer empathy/question signals → would lose long tail.

Pipeline:
  raw_samples.jsonl  →  [score 7-dim with v2.1 fixes]
                     →  [pair: max-score vs min-score, gap >= MIN_GAP_FOR_TAG]
                     →  [stratified 95/5 split by tag]
                     →  dpo_train.jsonl + dpo_val.jsonl  (ms-swift DPO format)
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DATA_DIR = Path("/mnt/workspace/psy-qwen-dpo/data")
RAW = DATA_DIR / "raw_samples.jsonl"
OUT_TRAIN = DATA_DIR / "dpo_train.jsonl"
OUT_VAL = DATA_DIR / "dpo_val.jsonl"
OUT_DEBUG = DATA_DIR / "scoring_debug.jsonl"

DEFAULT_MIN_GAP = 3.5
PER_TAG_MIN_GAP = {  # v2.2: with default 3.5, 心理学知识 needs even lower
    "心理学知识": 3.0,
}
VAL_RATIO = 0.05
SEED = 42

# Note: We do NOT define SYSTEM_PROMPT here — the real 738-char REBT system
# prompt is already embedded in each record's context (from PsyDTCorpus).
# The to_msswift_format() function preserves it as-is.

# ─────────────────────────────────────────────────────────────
# 7-dim rule-based scorer (v2.1)
# ─────────────────────────────────────────────────────────────
EMPATHY_WORDS = [
    "我理解", "我能感受到", "听起来", "看起来", "我注意到",
    "这一定", "想必", "或许", "也许", "似乎",
    "很正常", "可以理解", "是合理的", "是正常的",
    "不容易", "很辛苦", "很难受", "很痛苦",
    "难过", "焦虑", "无助", "沮丧", "困惑", "挫败",
    "愤怒", "委屈", "失落", "压力", "疲惫",
    "愿意", "可以多说", "想多了解", "想听你",
    "你的感受", "你的想法", "对你来说",
    "没关系", "不用着急", "慢慢来", "陪着你", "和你一起",
]

HARMFUL_PATTERNS = [
    r"你应该", r"你必须", r"你不应该", r"你不能",
    r"你要记住", r"你得",
    r"不要难过", r"别难过", r"不要哭", r"想开点",
    r"这没什么", r"别想那么多",
    r"建议你立刻", r"赶紧去", r"马上停止",
    r"我不是专业的", r"我没法帮你",
]

TEMPLATE_INDICATORS = [
    ["我理解", "我能理解"],
    ["这是正常的", "很正常", "是合理的"],
    ["尝试", "试着", "可以试", "建议"],
]


def score_response(text: str) -> dict:
    s = {"empathy": 0.0, "length": 0.0, "harmful": 0.0,
         "empathy_first": 0.0, "repetition": 0.0,
         "good_question": 0.0, "template_penalty": 0.0}

    if not text or not text.strip():
        s["total"] = -10.0
        return s

    # Dim 1: Empathy density
    hits = sum(1 for w in EMPATHY_WORDS if w in text)
    s["empathy"] = min(hits * 0.5, 4.0)

    # Dim 2: Length (v2.1: looser floor, 15-30 gets +0.5 instead of 0)
    L = len(text)
    if 50 <= L <= 300:
        s["length"] = 2.0
    elif 30 <= L < 50:
        s["length"] = 1.0
    elif 15 <= L < 30:           # ⬆️ now +0.5 instead of 0
        s["length"] = 0.5
    elif L < 15:                 # ⬇️ from <20 to <15
        s["length"] = -2.0
    else:                        # L > 300
        s["length"] = -1.0

    # Dim 3: Harmful patterns
    harmful_hits = sum(1 for p in HARMFUL_PATTERNS if re.search(p, text))
    s["harmful"] = -1.5 * harmful_hits

    # Dim 6 (computed early, drives Dim 4 logic): Good open-ended question
    has_q = "?" in text or "？" in text
    addresses_client = "你" in text or "您" in text
    is_good_question = False
    if has_q and addresses_client:
        if L < 80:
            s["good_question"] = 1.5
            is_good_question = True
        elif L < 150:
            s["good_question"] = 0.5
            # Note: medium-length questions don't trigger empathy_first exemption

    # Dim 4: Empathy-first heuristic (v2.1: exempt good questions)
    first_third = text[: max(1, len(text) // 3)]
    if any(w in first_third for w in EMPATHY_WORDS):
        s["empathy_first"] = 1.5
    elif is_good_question:        # ⬆️ NEW: pure questions don't need empathy opener
        s["empathy_first"] = 0.0
    else:
        s["empathy_first"] = -1.0

    # Dim 5: 4-gram repetition penalty
    if len(text) >= 4:
        grams = [text[i:i + 4] for i in range(len(text) - 3)]
        gram_counts = Counter(grams)
        repeated = sum(1 for c in gram_counts.values() if c >= 3)
        s["repetition"] = -0.5 * repeated

    # Dim 7: Template-empathy penalty
    groups_hit = 0
    for group in TEMPLATE_INDICATORS:
        if any(ind in text for ind in group):
            groups_hit += 1
    if groups_hit >= 2:
        s["template_penalty"] = -1.0

    s["total"] = sum(v for k, v in s.items() if k != "total")
    return s


# ─────────────────────────────────────────────────────────────
# Build pairs
# ─────────────────────────────────────────────────────────────
def build_pair(record: dict) -> dict | None:
    samples = record["samples"]
    scored = [(s, score_response(s)) for s in samples]
    scored.sort(key=lambda x: x[1]["total"], reverse=True)

    best_text, best_s = scored[0]
    worst_text, worst_s = scored[-1]
    gap = best_s["total"] - worst_s["total"]

    # v2.1: per-tag MIN_GAP
    min_gap = PER_TAG_MIN_GAP.get(record["tag"], DEFAULT_MIN_GAP)
    if gap < min_gap:
        return None

    return {
        "prompt_id": record["prompt_id"],
        "tag": record["tag"],
        "context": record["context"],
        "chosen": best_text,
        "rejected": worst_text,
        "chosen_score": round(best_s["total"], 2),
        "rejected_score": round(worst_s["total"], 2),
        "gap": round(gap, 2),
        "all_scores": [round(s[1]["total"], 2) for s in scored],
    }


def to_msswift_format(pair: dict) -> dict:
    """
    Note: pair['context'] already contains the system message from PsyDTCorpus
    (the 738-char REBT prompt). We do NOT prepend a duplicate SYSTEM_PROMPT.
    """
    messages = list(pair["context"])  # already has [system, user, assistant, user, ...]
    messages.append({"role": "assistant", "content": pair["chosen"]})
    return {
        "messages": messages,
        "rejected_response": pair["rejected"],
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)

    records = []
    with open(RAW, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"📥 Loaded {len(records)} prompts × 4 samples = {len(records) * 4} responses")
    print(f"⚙️  v2.1 scorer | DEFAULT_MIN_GAP={DEFAULT_MIN_GAP} | per-tag overrides: {PER_TAG_MIN_GAP}")

    pairs = []
    skipped = 0
    debug_rows = []
    for r in records:
        scored = [(s, score_response(s)) for s in r["samples"]]
        for text, s in scored:
            debug_rows.append({
                "prompt_id": r["prompt_id"],
                "tag": r["tag"],
                "text": text,
                **s,
            })

        pair = build_pair(r)
        if pair is None:
            skipped += 1
        else:
            pairs.append(pair)

    print(f"✅ Built {len(pairs)} pairs (skipped {skipped})")

    chosen_scores = [p["chosen_score"] for p in pairs]
    rejected_scores = [p["rejected_score"] for p in pairs]
    chosen_lens = [len(p["chosen"]) for p in pairs]
    rejected_lens = [len(p["rejected"]) for p in pairs]
    print(f"  chosen score   avg = {sum(chosen_scores)/len(chosen_scores):+.2f}")
    print(f"  rejected score avg = {sum(rejected_scores)/len(rejected_scores):+.2f}")
    print(f"  avg gap = {sum(p['gap'] for p in pairs)/len(pairs):+.2f}")
    print(f"  chosen avg length   = {sum(chosen_lens)/len(chosen_lens):.1f} chars")
    print(f"  rejected avg length = {sum(rejected_lens)/len(rejected_lens):.1f} chars")
    print(f"  length diff (chosen - rejected) = {sum(chosen_lens)/len(chosen_lens) - sum(rejected_lens)/len(rejected_lens):+.1f} chars")

    tag_dist = Counter(p["tag"] for p in pairs)
    print(f"\n📊 Pairs per tag:")
    for tag, n in sorted(tag_dist.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {n}")

    # Stratified train/val split
    by_tag = defaultdict(list)
    for p in pairs:
        by_tag[p["tag"]].append(p)
    train, val = [], []
    for tag, ps in by_tag.items():
        random.shuffle(ps)
        n_val = max(1, int(len(ps) * VAL_RATIO))
        val.extend(ps[:n_val])
        train.extend(ps[n_val:])
    random.shuffle(train)
    random.shuffle(val)
    print(f"\n🔀 Split: {len(train)} train + {len(val)} val")

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps(to_msswift_format(p), ensure_ascii=False) + "\n")
    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for p in val:
            f.write(json.dumps(to_msswift_format(p), ensure_ascii=False) + "\n")
    with open(OUT_DEBUG, "w", encoding="utf-8") as f:
        for row in debug_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n💾 Wrote:")
    print(f"  {OUT_TRAIN}  ({OUT_TRAIN.stat().st_size / 1024:.1f} KB)")
    print(f"  {OUT_VAL}    ({OUT_VAL.stat().st_size / 1024:.1f} KB)")
    print(f"  {OUT_DEBUG}  ({OUT_DEBUG.stat().st_size / 1024:.1f} KB)")

    # Show 3 sample pairs
    print("\n" + "=" * 60)
    print("🔍 Sample pairs (sorted by gap, showing 3):")
    print("=" * 60)
    pairs.sort(key=lambda x: -x["gap"])
    for p in pairs[:3]:
        print(f"\n--- prompt_id={p['prompt_id']} | tag={p['tag']} | gap={p['gap']} ---")
        print(f"  CHOSEN   ({p['chosen_score']}): {p['chosen'][:150]}")
        print(f"  REJECTED ({p['rejected_score']}): {p['rejected'][:150]}")
        print(f"  all 4 scores: {p['all_scores']}")


if __name__ == "__main__":
    main()