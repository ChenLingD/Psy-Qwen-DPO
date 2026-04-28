# Psy-Qwen-DPO 项目调试日志

## 重启项目时遇到的环境问题（2026-04-26）

### 问题 1: peft import HybridCache 错误
**症状**: `ImportError: cannot import name 'HybridCache' from 'transformers'`
**根因**: 魔塔实例镜像自动升级 transformers 到 5.3.0，但 peft 0.17.1 与之不兼容（peft < 0.18.0 不支持 transformers v5）
**解决**: `pip install "transformers>=5.2.0,<5.4.0" "peft>=0.18.0"`
**最终版本**: transformers 5.3.0 + peft 0.19.1

### 问题 2: model_type 'qwen3_5' 不识别
**症状**: `ValueError: The checkpoint you are trying to load has model type 'qwen3_5' but Transformers does not recognize this architecture`
**根因**: 我曾尝试降级 transformers 到 4.46.3，但 Qwen3.5 架构需要 transformers >= 5.2.0 才支持
**解决**: 升级回 transformers 5.3.0
**经验**: 不要凭模型发布日期猜版本，要看 args.json 里训练时实际记录的依赖

### 问题 3: LoRA target_modules not found
**症状**: `ValueError: Target modules ^(model\.language_model...)$ not found in the base model`
**根因**: ms-swift 训练 SFT 时把 Qwen3.5-0.8B 当 VLM 处理（多模态架构），自动给 target_modules 加了 'model.language_model.' 前缀；但用 AutoModelForCausalLM 加载时模型是纯语言模型结构，没有 language_model 这一层
**解决**: 修改 adapter_config.json，把 target_modules 中的 `model\.language_model` 替换为 `model`
**最终路径**: `^(model(?=\.).*\.(in_proj_qkv|gate_proj|up_proj|...))$`

### 问题 4: 推理时模型自言自语 + 退化重复
**症状**: 模型生成完一轮后，自动续写 user/assistant 多轮对话，并出现"沮丧，沮丧，沮丧"重复词
**初看像是**: 模型训练有问题
**真实根因**: generate() 调用没显式传 eos_token_id 参数，默认用 pad_token_id (`<|endoftext|>` 248044) 当 stop token，但模型实际生成的 stop token 是 `<|im_end|>` (248046)，两个 ID 不一致导致模型不停下来
**解决**: 在 generate() 中显式传 `eos_token_id=tokenizer.eos_token_id`
**经验**: 模型行为异常时，先怀疑生成参数 / chat template 这种"接口层"问题，再怀疑模型本身

## 关键经验总结

1. **环境记录优先于性能**：args.json 里的训练时依赖版本是真相之源，不要凭直觉猜
2. **报错信息要细读**：每个错误的关键词（HybridCache / qwen3_5 / target_modules / 重复退化）都直接指向了根因
3. **bug 优先怀疑外围**：模型行为异常时先查 chat template、generate 参数、tokenizer 配置，再查模型本身
4. **改文件先备份**：adapter_config.json 改之前留了 .bak，整个过程零风险

## 数据集深度分析（Task 1.1 / 1.2 阶段）

### 发现：PsyDTCorpus 是单疗法多主题数据集
- **总规模**：4760 多轮对话，平均 37.2 轮/对话，最长 71 轮
- **主题**：12 个心理咨询主题（婚恋 21.3% 最多，心理学知识 0.7% 最少）
- **疗法**：100% REBT（理情行为疗法），唯一 system prompt
- **System prompt 长度**：738 字符（固定）

### 含义
- DPO 项目可以聚焦于 "REBT 风格回复对齐"，不需处理跨疗法问题
- README 中应明确定位为 "REBT-specialized counseling LLM"，而不是泛化心理咨询

### 采样策略决策
- 采用 **分层采样 + 最低配额**（避免婚恋主题主导，保护长尾主题）
- 12 个主题目标分布：[200, 150, 150, 130, 120, 100, 95, 90, 60, 55, 50, 50] = 1250
- 实际采集 1232（心理学知识只有 34 条对话，过滤后 32 条）
- 截断点选对话中段（25-75% 之间），保留完整上下文
