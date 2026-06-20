# FareedKhan-dev/train-llm-from-scratch vs 本项目对比分析

> 2026-06-18

---

## 一、项目定位

| 维度 | FareedKhan-dev | 本项目 (4 projects) |
|------|---------------|-------------------|
| **定位** | LLM 全生命周期训练框架 | 多领域 ML 实战项目群 |
| **核心** | Transformer 从零 + RLHF | 金融/文本/保险 三领域 |
| **目标** | 训练自己的 LLM | 教程知识实践 + 领域应用 |
| **模型** | 30M Transformer (从零) | 0.5M LSTM + Qwen3 LoRA + LightGBM |
| **语言** | 纯 PyTorch (no trl/peft) | PyTorch + HF + LightGBM |

---

## 二、训练管道对比

### FareedKhan-dev 完整 LLM 管道

```
数据下载 → 预处理 → 预训练 → SFT → Reward Model → DPO → PPO → GRPO → 评估 → Chat
  (9 stages, fully scripted, Streamlit UI per stage)
```

### 本项目 4 项目管道

```
Kronos: OHLCV → BSQ Tokenizer → Transformer预训练 → LSTM回归 → RankIC评估
医疗:   MD文档 → Qwen3预训练 → LoRA续写 → ChatML指令 → 生成
法规:   MD文档 → Qwen3预训练 → LoRA续写 → 生成
控费:   CSV → LightGBM训练 → Optuna调优 → Walk-forward → 预测
```

---

## 三、阶段覆盖对比

| 训练阶段 | FareedKhan | Kronos | 医疗 | 法规 | 控费 |
|---------|-----------|--------|------|------|------|
| 数据预处理 | ✅ | ✅ OHLCV→CSV | ✅ MD清洗 | ✅ | ✅ polars |
| 预训练 | ✅ 30M Transformer | ✅ BSQ+Transformer | ✅ Qwen3预训练 | ✅ | — |
| 监督微调(SFT) | ✅ | — | ✅ 续写 | ✅ 续写 | — |
| 指令微调 | ✅ | — | ✅ ChatML | — | — |
| 奖励模型 | ✅ | — | — | — | — |
| DPO对齐 | ✅ | — | — | — | — |
| PPO RL | ✅ | — | — | — | — |
| GRPO RL | ✅ | — | — | — | — |
| 分类微调 | — | ✅ Ch6 | — | — | — |
| 超参优化 | — | 手动扫描 | — | — | ✅ Optuna |
| 蒙特卡洛检验 | — | ✅ | — | — | — |
| 模型评估 | ✅ | ✅ RankIC | — | — | ✅ MAE |
| Chat UI | ✅ Streamlit | — | — | — | — |

---

## 四、优势与差距

### FareedKhan-dev 优势 (本项目缺失)

| 能力 | 说明 |
|------|------|
| **RLHF 全管道** | DPO/PPO/GRPO 从零实现 (no trl) |
| **Streamlit UI** | 每阶段独立界面 |
| **配置驱动** | JSON 配置管理 |
| **完整文档** | 9篇详细文档 + 架构图 |
| **端到端** | data→chat 一站式 |

### 本项目优势 (FareedKhan-dev 缺失)

| 能力 | 说明 |
|------|------|
| **多领域覆盖** | 金融时序 + 文本生成 + 保险精算 |
| **BSQ 量化** | 将连续数据离散化为 token |
| **金融评估** | RankIC / Sharpe / Monte Carlo |
| **LoRA 微调** | 0.7M 参数极低资源 |
| **LightGBM** | 结构化数据 ML 最佳实践 |
| **生产部署** | Predictor API + 每日流水线 |
| **教程覆盖跟踪** | 93% 覆盖率量化评估 |

---

## 五、教程覆盖对比

| 章节 | FareedKhan | 本项目 |
|------|-----------|--------|
| Ch1 LLM概念 | ✅ 30M Transformer | ✅ BSQ+Transformer |
| Ch2 数据处理 | ✅ text pipeline | ✅ BSQ tokenizer |
| Ch3 注意力 | ✅ 从零实现 | ✅ 从零实现 + GQA |
| Ch4 GPT模型 | ✅ Transformer | ✅ Decoder-only |
| Ch5 预训练 | ✅ 完整循环 | ✅ 两阶段训练 |
| Ch6 分类微调 | — | ✅ Ch6 script |
| Ch7 指令微调 | ✅ SFT+DPO+PPO+GRPO | ✅ ChatML (医疗) |
| **RLHF (额外)** | **✅ DPO+PPO+GRPO** | ❌ 缺失 |
| App A PyTorch | ✅ | ✅ |
| App D 训练增强 | ✅ | ✅ Cosine+Clip+MC |
| App E LoRA | — | ✅ PEFT r=8 |

### 覆盖总结

| 项目 | 教程覆盖 | 额外覆盖 |
|------|---------|---------|
| FareedKhan | ~85% (缺Ch6分类微调, AppE LoRA) | **RLHF全管道** (DPO/PPO/GRPO) |
| 本项目 | **93%** | 金融预测/保险ML/蒙特卡洛检验 |

---

## 六、可互相借鉴

| 来源 → 目标 | 借鉴内容 |
|------------|---------|
| FareedKhan → 本项目 | **RLHF 管道**: DPO/PPO/GRPO 从零实现 |
| FareedKhan → 本项目 | **Streamlit UI**: 每阶段可视化 |
| FareedKhan → 本项目 | **配置管理**: JSON 配置替代硬编码 |
| 本项目 → FareedKhan | **Ch6 分类微调**: 替换输出头 + 冻结策略 |
| 本项目 → FareedKhan | **LoRA 微调**: 0.7M 极低资源方案 |
| 本项目 → FareedKhan | **蒙特卡洛检验**: 统计显著性验证 |
| 本项目 → FareedKhan | **跨领域应用**: 金融/保险模型 |

---

## 七、结论

| 维度 | FareedKhan-dev | 本项目 |
|------|---------------|--------|
| 定位 | **LLM训练框架** (一站式) | **多领域ML项目群** (教程实践) |
| 深度 | RLHF 全实现 | 跨领域广度 |
| 教程覆盖 | ~85% | **93%** |
| 独有优势 | DPO/PPO/GRPO + Streamlit UI | 金融RankIC + 保险ML + Monte Carlo |
| 最值得借鉴 | RLHF管道 | 分类微调 + LoRA + 统计检验 |

两个项目互补性强：本项目覆盖教程更多章节且有跨领域实践，FareedKhan 在 RLHF 后训练管道上有完整实现。
