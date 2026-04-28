"""
修复 fig2 和 fig3：
- fig2: 把右图的 final eval 文字框移到左上角（避开 legend）
- fig3: 把 tag 名翻译成英文 + 调整 layout 不让上图 legend 撞到下图 title
"""

import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

PROJECT_DIR = Path("/mnt/workspace/psy-qwen-dpo")
TRAINER_STATE = Path(
    "/mnt/workspace/output/psy-qwen-dpo-beta01/v1-20260428-033426/"
    "checkpoint-162/trainer_state.json"
)
RESULTS_FILE = PROJECT_DIR / "data" / "phase3_results.json"
OUT_DIR = PROJECT_DIR / "outputs" / "figures"

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

COLOR_DPO = "#2E5EAA"
COLOR_SFT = "#E8973A"
COLOR_NEUTRAL = "#888888"
COLOR_HIGHLIGHT = "#3CB371"

# 中→英 tag 翻译表
TAG_TRANSLATIONS = {
    "婚恋": "Romance",
    "人际": "Interpersonal",
    "家庭": "Family",
    "治疗": "Therapy",
    "情绪": "Emotion",
    "成长": "Growth",
    "自我": "Self",
    "行为": "Behavior",
    "社会": "Society",
    "职场": "Workplace",
    "性心理": "Sexuality",
    "心理学知识": "Psych Knowledge",
}


# ============ Fig 2: 训练曲线（修文字框位置） ============
def make_fig2():
    with open(TRAINER_STATE) as f:
        state = json.load(f)

    train_log = [e for e in state["log_history"] if "loss" in e and "eval_loss" not in e]
    eval_log = [e for e in state["log_history"] if "eval_loss" in e]

    train_steps = [e["step"] for e in train_log]
    train_loss = [e["loss"] for e in train_log]
    train_margins = [e["rewards/margins"] for e in train_log]

    eval_steps = [e["step"] for e in eval_log]
    eval_loss = [e["eval_loss"] for e in eval_log]
    eval_margins = [e["eval_rewards/margins"] for e in eval_log]
    eval_acc = [e["eval_rewards/accuracies"] for e in eval_log]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左图
    ax = axes[0]
    ax.plot(train_steps, train_loss, color=COLOR_DPO, linewidth=2,
            label="train loss", marker="o", markersize=3)
    ax.plot(eval_steps, eval_loss, color=COLOR_HIGHLIGHT, linewidth=2,
            label="eval loss", marker="s", markersize=8, linestyle="--")
    ax.axhline(0.6931, color="gray", linestyle=":", alpha=0.6,
               label="initial loss (ln 2 ≈ 0.693)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("DPO loss")
    ax.set_title("Loss Curves: 0.693 → 0.378 (−45%)", fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.annotate(f"final eval = {eval_loss[-1]:.3f}",
                xy=(eval_steps[-1], eval_loss[-1]),
                xytext=(eval_steps[-1] - 35, eval_loss[-1] + 0.05),
                fontsize=10, color=COLOR_HIGHLIGHT,
                arrowprops=dict(arrowstyle="->", color=COLOR_HIGHLIGHT))

    # 右图：把文字框移到左上，legend 移到右下
    ax = axes[1]
    ax.plot(train_steps, train_margins, color=COLOR_DPO, linewidth=2,
            label="train margin", marker="o", markersize=3)
    ax.plot(eval_steps, eval_margins, color=COLOR_HIGHLIGHT, linewidth=2,
            label="eval margin", marker="s", markersize=8, linestyle="--")
    ax.set_xlabel("Training step")
    ax.set_ylabel("rewards/margins (chosen − rejected)")
    ax.set_title(f"Reward Margin Growth → final {eval_acc[-1]*100:.1f}% accuracy",
                 fontweight="bold")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5, linewidth=0.8)
    # 文字框移到左上角
    ax.text(0.03, 0.97,
            f"Final eval @ step {eval_steps[-1]}:\n"
            f"  margin = {eval_margins[-1]:.3f}\n"
            f"  accuracy = {eval_acc[-1]*100:.2f}%",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=COLOR_HIGHLIGHT, lw=1.2))
    # legend 移到右下角（与文字框分开）
    ax.legend(loc="lower right", frameon=False)

    plt.tight_layout()
    out = OUT_DIR / "fig2_training_curves.png"
    plt.savefig(out)
    plt.close()
    print(f"✅ {out.name}")


# ============ Fig 3: Win rate（英文 tag + 修 layout） ============
def make_fig3():
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    # 用 GridSpec，留更多 hspace
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.4], hspace=0.35)

    # 上：堆叠条
    ax1 = fig.add_subplot(gs[0])
    cb = results["category_breakdown"]
    categories = [
        ("DPO consistent win",  cb.get("consistent_dpo_win", 0),  COLOR_DPO),
        ("DPO win + tie",       cb.get("dpo_win_tie", 0),         "#5C8AC8"),
        ("Both tie",            cb.get("consistent_tie", 0),      "#BBBBBB"),
        ("Position bias (1-1)", cb.get("inconsistent_position_bias", 0), "#888888"),
        ("SFT win + tie",       cb.get("sft_win_tie", 0),         "#F0B673"),
        ("SFT consistent win",  cb.get("consistent_sft_win", 0),  COLOR_SFT),
    ]
    total = results["total_pairs"]
    left = 0
    for label, n, color in categories:
        if n == 0: continue
        ax1.barh([0], [n], left=left, color=color, label=f"{label} ({n})",
                 edgecolor="white", linewidth=1.5, height=0.5)
        if n / total > 0.04:
            ax1.text(left + n / 2, 0, f"{n}", ha="center", va="center",
                     fontweight="bold", color="white", fontsize=11)
        left += n
    ax1.set_xlim(0, total)
    ax1.set_yticks([])
    ax1.set_xlabel(f"Number of held-out prompts (total = {total})")
    ax1.set_title(
        f"Phase 3 LLM-Judge Evaluation: Win Rate = {results['win_rate_pct']}%   "
        f"(judge = DeepSeek V4-Flash, 2-way position-bias mitigation)",
        fontweight="bold"
    )
    # legend 改成 2 行 ncol=3，向下偏移更多
    ax1.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
               ncol=1, frameon=False, fontsize=9)
    ax1.grid(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)

    # 下：tag 横向柱状（英文）
    ax2 = fig.add_subplot(gs[1])
    by_tag = results["by_tag"]
    sorted_tags = sorted(by_tag.items(), key=lambda x: x[1]["win_rate_pct"])
    tag_names = [
        f"{TAG_TRANSLATIONS.get(tag, tag)} (n={info['n']})"
        for tag, info in sorted_tags
    ]
    tag_wr = [info["win_rate_pct"] for _, info in sorted_tags]

    bar_colors = [COLOR_DPO if wr >= 70 else "#6B95D1" if wr >= 60
                  else COLOR_SFT for wr in tag_wr]

    bars = ax2.barh(tag_names, tag_wr, color=bar_colors,
                    edgecolor="white", linewidth=1.2)
    ax2.axvline(50, color="gray", linestyle=":", alpha=0.6, label="50% (random)")
    ax2.axvline(results["win_rate_pct"], color=COLOR_HIGHLIGHT,
                linestyle="--", linewidth=2,
                label=f"overall {results['win_rate_pct']}%")
    ax2.set_xlabel("DPO win rate (%)")
    ax2.set_title("Win Rate by Psychological Topic (12 tags)", fontweight="bold")
    ax2.set_xlim(0, 110)
    ax2.legend(loc="lower right", frameon=False, fontsize=9)
    for bar, v in zip(bars, tag_wr):
        ax2.text(v + 1.5, bar.get_y() + bar.get_height() / 2,
                 f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")

    out = OUT_DIR / "fig3_win_rate.png"
    plt.savefig(out)
    plt.close()
    print(f"✅ {out.name}")


def main():
    make_fig2()
    make_fig3()
    print(f"\n✅ Fixed figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()