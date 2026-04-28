#!/bin/bash
# ============================================================================
# Psy-Qwen-DPO 训练脚本（方案 B：β=0.1 单组）
# ============================================================================
# 创建时间: 2026-04-27
# 基础: ms-swift 4.1.0.dev0 + peft 0.19.1 + transformers 5.3.0 + torch 2.9.1
# 硬件: NVIDIA A10 24GB
#
# 数据资产:
#   train: /mnt/workspace/psy-qwen-dpo/data/dpo_train.jsonl  (431 pairs)
#   val:   /mnt/workspace/psy-qwen-dpo/data/dpo_val.jsonl    (19 pairs)
#
# 模型起点:
#   Base:       Qwen3.5-0.8B (~0.8B params, frozen during DPO)
#   SFT LoRA:   v2-20260408 checkpoint-375 (rank=8, alpha=32, ~5.4M params)
#
# 预期产出:
#   /mnt/workspace/output/psy-qwen-dpo-beta01/
#     └── checkpoint-XXX/  (DPO LoRA, ~21MB adapter_model.safetensors)
#
# 预期时间:
#   3 epochs × 431 pairs / (1 batch × 8 grad_accum) ≈ 162 steps
#   单步约 30-60s on A10 → 总计约 1.5-3 小时（与序列长度相关）
# ============================================================================

set -euo pipefail   # 任何命令失败立即退出，避免静默错误

# ----------- 路径配置 -----------
BASE_MODEL="/mnt/workspace/.cache/modelscope/models/Qwen/Qwen3___5-0___8B"
SFT_LORA="/mnt/workspace/output/psy-qwen-0.8b/v2-20260408-061615/checkpoint-375"
TRAIN_DATA="/mnt/workspace/psy-qwen-dpo/data/dpo_train.jsonl"
VAL_DATA="/mnt/workspace/psy-qwen-dpo/data/dpo_val.jsonl"
OUTPUT_DIR="/mnt/workspace/output/psy-qwen-dpo-beta01"

# SFT 时的 LoRA target_modules 正则（必须与 SFT 完全一致）
TARGET_REGEX='^(model(?=\.).*\.(in_proj_qkv|gate_proj|up_proj|in_proj_b|q_proj|out_proj|v_proj|in_proj_z|in_proj_a|down_proj|o_proj|k_proj))$'

# ----------- 训练超参 -----------
mkdir -p "${OUTPUT_DIR}"

swift rlhf \
    --rlhf_type dpo \
    --model "${BASE_MODEL}" \
    --adapters "${SFT_LORA}" \
    --ref_adapters "${SFT_LORA}" \
    \
    --dataset "${TRAIN_DATA}" \
    --val_dataset "${VAL_DATA}" \
    --max_length 1024 \
    --truncation_strategy left \
    \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_bias none \
    --target_regex "${TARGET_REGEX}" \
    \
    --beta 0.1 \
    --loss_type sigmoid \
    \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    \
    --bf16 true \
    --gradient_checkpointing true \
    --torch_dtype bfloat16 \
    \
    --logging_steps 5 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --eval_strategy epoch \
    --report_to none \
    --seed 42 \
    \
    --output_dir "${OUTPUT_DIR}" \
    --logging_dir "${OUTPUT_DIR}/logs"

# ============================================================================
# 各参数的"为什么":
#
# [模型加载]
#   --rlhf_type dpo            选 DPO 算法（其他: orpo/simpo/kto/cpo/rm/ppo/grpo/gkd）
#   --model BASE_MODEL         加载冻结的 Qwen3.5-0.8B base
#   --adapters SFT_LORA        在 base 上叠加 SFT LoRA → policy 的起点
#   --ref_adapters SFT_LORA    再叠加一份**同样的** SFT LoRA → reference（冻结）
#                              policy 和 reference 初始权重相同，训练后 policy 漂移
#
# [数据]
#   --max_length 1024          覆盖 PsyDTCorpus 99% 长度，OOM 时降到 768
#   --truncation_strategy left 超长时从左侧（早期对话）截断，保留近期 context
#
# [LoRA]  全部与 SFT 一致，DPO 必须保持
#   --lora_rank 8, --lora_alpha 32, --lora_dropout 0.05, --lora_bias none
#   --target_regex 正则匹配所有 attention/MLP projection
#
# [DPO 算法]
#   --beta 0.1                 KL 约束系数（论文起点）。
#                              大 → 保守，小 → 激进。
#   --loss_type sigmoid        DPO 标准 loss
#
# [训练]
#   --num_train_epochs 3       431 × 3 / 8 ≈ 162 步
#   --learning_rate 5e-6       DPO 比 SFT 小 5-10x（SFT 时是 1e-4）
#   --warmup_ratio 0.05        前 5% step warmup
#   --max_grad_norm 1.0        DPO 偶有 loss spike，必裁剪
#
# [精度]
#   --bf16 + --gradient_checkpointing  双模型省显存必开
#
# [日志]
#   --logging_steps 5          关键看 rewards/margins 是否在涨
#   --save_strategy epoch      每 epoch 末存（共 3 个 ckpt）
# ============================================================================