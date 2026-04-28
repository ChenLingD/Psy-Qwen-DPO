"""
View dpo pairs in human-friendly format for error analysis.

Usage:
  python scripts/view_pairs.py            # show 30 random pairs
  python scripts/view_pairs.py --n 50     # show 50 pairs
  python scripts/view_pairs.py --gap-min 3 --gap-max 5   # only mid-gap pairs
  python scripts/view_pairs.py --out review.txt          # write to file
"""
import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path("/mnt/workspace/psy-qwen-dpo/data")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--gap-min", type=float, default=0.0)
    ap.add_argument("--gap-max", type=float, default=999.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None,
                    help="If set, also write the formatted output to this file")
    args = ap.parse_args()

    # Load debug file (has all 4 scores per prompt) AND train file (has chosen/rejected)
    # We work backwards: load debug, group by prompt_id, find the highest+lowest pair
    debug_path = DATA_DIR / "scoring_debug.jsonl"
    rows = []
    with open(debug_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # Group by prompt_id
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[r["prompt_id"]].append(r)

    # Build pairs (replay scoring logic — top vs bottom)
    pairs = []
    for pid, items in groups.items():
        items.sort(key=lambda x: x["total"], reverse=True)
        gap = items[0]["total"] - items[-1]["total"]
        if gap < 3.0:  # same MIN_GAP as score_and_pair.py
            continue
        if not (args.gap_min <= gap <= args.gap_max):
            continue
        pairs.append({
            "prompt_id": pid,
            "tag": items[0]["tag"],
            "chosen": items[0]["text"],
            "rejected": items[-1]["text"],
            "chosen_score": items[0]["total"],
            "rejected_score": items[-1]["total"],
            "gap": gap,
            "all_scores": [round(x["total"], 2) for x in items],
            # Per-dim breakdown for chosen and rejected
            "chosen_breakdown": {k: round(items[0][k], 2) for k in
                                 ["empathy", "length", "harmful", "empathy_first", "repetition"]},
            "rejected_breakdown": {k: round(items[-1][k], 2) for k in
                                   ["empathy", "length", "harmful", "empathy_first", "repetition"]},
        })

    random.seed(args.seed)
    random.shuffle(pairs)
    pairs = pairs[: args.n]

    # We also need the user prompt context — load from raw_samples.jsonl
    raw_path = DATA_DIR / "raw_samples.jsonl"
    raw_by_id = {}
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            raw_by_id[r["prompt_id"]] = r

    # Format
    lines = []
    for i, p in enumerate(pairs, 1):
        raw = raw_by_id.get(p["prompt_id"], {})
        ctx = raw.get("context", [])
        # Last user message is the actual prompt the model is responding to
        last_user = next((m["content"] for m in reversed(ctx) if m["role"] == "user"), "(no user msg)")

        lines.append(f"\n{'='*70}")
        lines.append(f"# Pair {i}/{len(pairs)}  |  prompt_id={p['prompt_id']}  |  tag={p['tag']}  |  gap={p['gap']:.2f}")
        lines.append(f"{'='*70}")
        lines.append(f"用户最后一句: {last_user[:200]}")
        lines.append(f"")
        lines.append(f"[CHOSEN  score={p['chosen_score']:+.2f}]  {p['chosen_breakdown']}")
        lines.append(f"  → {p['chosen']}")
        lines.append(f"")
        lines.append(f"[REJECTED score={p['rejected_score']:+.2f}]  {p['rejected_breakdown']}")
        lines.append(f"  → {p['rejected']}")
        lines.append(f"")
        lines.append(f"all 4 scores: {p['all_scores']}")
        lines.append(f"")
        lines.append(f"我的判断: [ ] chosen 真的更好  [ ] 差不多  [ ] rejected 其实更好/差不多")
        lines.append(f"如果是 ❌：是因为 [ ] chosen 虚高  [ ] rejected 被冤枉  [ ] 都有")

    text = "\n".join(lines)
    print(text)

    if args.out:
        out_path = Path(args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n✅ 已写入 {out_path}")


if __name__ == "__main__":
    main()