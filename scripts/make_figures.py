"""
Phase 4.1: 生成 README 用的 3 张关键图
- fig1_scorer_iteration.png: Scorer 迭代 before/after
- fig2_training_curves.png: DPO 训练曲线（loss + rewards/margins）
- fig3_win_rate.png: Phase 3 win rate 总体 + 按 tag 分解
"""

import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ============ 配置 ============
PROJECT_DIR = Path("/mnt/workspace/psy-qwen-dpo")
TRAINER_STATE = Path(
    "/mnt/workspace/output/psy-qwen-dpo-beta01/v1-20260428-033426/"
    "checkpoint-162/trainer_state.json"
)
RESULTS_FILE = PROJECT_DIR / "data" / "phase3_results.json"
OUT_DIR = PROJECT_DIR / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ 全局样式 ============
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# 配色
COLOR_DPO = "#2E5EAA"   # 深蓝 = DPO
COLOR_SFT = "#E8973A"   # 橙色 = SFT
COLOR_NEUTRAL = "#888888"
COLOR_HIGHLIGHT = "#3CB371"  # 绿色（强调）


# ============ 图 1: Scorer 迭代 ============
def make_fig1_scorer():
    """v1 (43%) vs v2.2 (60%) 双柱图，附 length bias 改善"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- 左：chosen-better 准确率 ----
    versions = ["Scorer v1\n(initial)", "Scorer v2.2\n(after 2 iterations)"]
    accuracy = [43, 60]
    bars = ax1.bar(versions, accuracy, color=[COLOR_NEUTRAL, COLOR_HIGHLIGHT],
                   width=0.55, edgecolor="white", linewidth=1.5)
    ax1.set_ylabel("Chosen-better agreement rate (%)")
    ax1.set_title("Scorer Quality: +17 pp via Error Analysis", fontweight="bold")
    ax1.set_ylim(0, 75)
    ax1.axhline(50, color="gray", linestyle=":", alpha=0.5, label="random baseline")
    ax1.legend(loc="upper left", frameon=False)
    for bar, v in zip(bars, accuracy):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                 f"{v}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
    # 箭头标注提升
    ax1.annotate("", xy=(1, 60), xytext=(0, 43),
                 arrowprops=dict(arrowstyle="->", color=COLOR_HIGHLIGHT, lw=2))
    ax1.text(0.5, 53, "+17 pp", ha="center", color=COLOR_HIGHLIGHT,
             fontweight="bold", fontsize=12)

    # ---- 右：length bias 改善 ----
    stages = ["Scorer v1", "Scorer v2.2", "Final DPO output\n(Phase 3 eval)"]
    bias = [12.7, 6.4, 1.3]
    bars = ax2.bar(stages, bias, color=[COLOR_NEUTRAL, "#FFA500", COLOR_HIGHLIGHT],
                   width=0.55, edgecolor="white", linewidth=1.5)
    ax2.set_ylabel("Length bias (chars, DPO − SFT)")
    ax2.set_title("Length Bias Reduction: 12.7 → 1.3 chars", fontweight="bold")
    ax2.set_ylim(0, 15)
    for bar, v in zip(bars, bias):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                 f"+{v}", ha="center", va="bottom", fontweight="bold", fontsize=12)

    plt.tight_layout()
    out = OUT_DIR / "fig1_scorer_iteration.png"
    plt.savefig(out)
    plt.close()
    print(f"✅ {out.name}")


# ============ 图 2: DPO 训练曲线 ============
def make_fig2_training():
    """双 y 轴：loss (左) + rewards/margins (右)"""
    with open(TRAINER_STATE) as f:
        state = json.load(f)

    train_log = [e for e in state["log_history"] if "loss" in e and "eval_loss" not in e]
    eval_log = [e for e in state["log_history"] if "eval_loss" in e]

    train_steps = [e["step"] for e in train_log]
    train_loss = [e["loss"] for e in train_log]
    train_margins = [e["rewards/margins"] for e in train_log]
    train_acc = [e["rewards/accuracies"] for e in train_log]

    eval_steps = [e["step"] for e in eval_log]
    eval_loss = [e["eval_loss"] for e in eval_log]
    eval_margins = [e["eval_rewards/margins"] for e in eval_log]
    eval_acc = [e["eval_rewards/accuracies"] for e in eval_log]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ---- 左图：Loss curves ----
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
    # 标注最终 eval loss
    ax.annotate(f"final eval = {eval_loss[-1]:.3f}",
                xy=(eval_steps[-1], eval_loss[-1]),
                xytext=(eval_steps[-1] - 35, eval_loss[-1] + 0.05),
                fontsize=10, color=COLOR_HIGHLIGHT,
                arrowprops=dict(arrowstyle="->", color=COLOR_HIGHLIGHT))

    # ---- 右图：Rewards margins + accuracy ----
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
    ax.legend(loc="lower right", frameon=False)
    # 在右上角加最终 accuracy 文字框
    ax.text(0.97, 0.05,
            f"Final eval @ step {eval_steps[-1]}:\n"
            f"  margin = {eval_margins[-1]:.3f}\n"
            f"  accuracy = {eval_acc[-1]*100:.2f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=COLOR_HIGHLIGHT, lw=1.2))

    plt.tight_layout()
    out = OUT_DIR / "fig2_training_curves.png"
    plt.savefig(out)
    plt.close()
    print(f"✅ {out.name}")


# ============ 图 3: Win Rate 分解 ============
def make_fig3_winrate():
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.45)

    # ---- 上：总体类别堆叠水平条 ----
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
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4),
               ncol=3, frameon=False, fontsize=9)
    ax1.grid(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)

    # ---- 下：按 tag 横向柱状 ----
    ax2 = fig.add_subplot(gs[1])
    by_tag = results["by_tag"]
    # 按 win rate 排序
    sorted_tags = sorted(by_tag.items(), key=lambda x: x[1]["win_rate_pct"])
    tag_names = [f"{tag} (n={info['n']})" for tag, info in sorted_tags]
    tag_wr = [info["win_rate_pct"] for _, info in sorted_tags]

    # 颜色映射：>=70% 深蓝，60-70% 中蓝，<60% 橙色（提醒）
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
    ax2.set_xlim(0, 105)
    ax2.legend(loc="lower right", frameon=False, fontsize=9)
    for bar, v in zip(bars, tag_wr):
        ax2.text(v + 1.5, bar.get_y() + bar.get_height() / 2,
                 f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")

    out = OUT_DIR / "fig3_win_rate.png"
    plt.savefig(out)
    plt.close()
    print(f"✅ {out.name}")


# ============ 主流程 ============
def main():
    print(f"Output dir: {OUT_DIR}\n")
    make_fig1_scorer()
    make_fig2_training()
    make_fig3_winrate()
    print(f"\n✅ All 3 figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()