# 三项目训练脚本对比分析

> 2026-06-17

---

## 概览

| 维度 | Kronos Stock | 医疗文本 | 法规文本 |
|------|-------------|---------|---------|
| **模型** | Kronos(BSQ+Transformer) + LSTM | Qwen3-0.6B + LoRA | Qwen3-0.6B + LoRA |
| **训练方式** | 两阶段(自编码器→自回归) | 单阶段续写微调 | 单阶段续写微调 |
| **参数量** | 0.5M~4.7M | 0.7M (LoRA) | 0.7M (LoRA) |
| **数据格式** | OHLCV时序/滑动窗口 | 纯文本(.md) | 纯文本(.md) |
| **数据集** | StockDataset(预计算索引) | MedicalTextDataset | RegulationDataset |
| **损失函数** | MSE(回归) / BCE(分类) | CrossEntropy(续写) | CrossEntropy(续写) |
| **训练脚本** | train_tokenizer.py + train_predictor.py | train_qwen_lora.py (423行) | train_qwen_lora.py (329行) |
| **训练行数** | 253+290+125=668行 | 423行 | 329行 |

---

## 详细对比

### 1. 断点续传

| 特性 | Kronos | 医疗 | 法规 |
|------|--------|------|------|
| 保存内容 | model+opt+scheduler+logs+elapsed | model+opt+scheduler+logs+elapsed | model+opt+scheduler+logs |
| 保存频率 | 每epoch覆盖 | 每epoch覆盖 | 每epoch覆盖 |
| 恢复方式 | 自动检测 checkpoint/training_state.pt | 自动检测 checkpoint/training_state.pt | 自动检测 checkpoint/training_state.pt |
| 一致性 | ✅ 三项目统一模式 | ✅ | ✅ |

### 2. 优化器与调度器

| 特性 | Kronos | 医疗 | 法规 |
|------|--------|------|------|
| 优化器 | AdamW | AdamW | AdamW |
| 调度器 | Cosine+Warmup | Linear Warmup | Cosine+Warmup |
| 梯度裁剪 | clip=5.0 | clip=1.0 | clip=1.0 |

### 3. 数据集设计

| 特性 | Kronos | 医疗 | 法规 |
|------|--------|------|------|
| 类型 | 时序滑动窗口 | 文本续写 | 文本续写 |
| 采样方式 | 随机采样(预计算索引) | 随机采样 | 随机采样 |
| 归一化 | Z-score(按窗口) | 无(Tokenizer处理) | 无(Tokenizer处理) |
| 多数据集 | 支持(val/test分离) | 支持(val/test分离) | 单一数据 |

### 4. 模型加载

| 特性 | Kronos | 医疗 | 法规 |
|------|--------|------|------|
| 方式 | 从零初始化 | HuggingFace预训练 | HuggingFace预训练 |
| 微调方式 | 全参数训练 | LoRA(0.12%参数) | LoRA(0.12%参数) |
| 续训支持 | ✅ --resume | ✅ --resume_from(PEFT) | ✅ checkpoint |

### 5. 评估方式

| 特性 | Kronos | 医疗 | 法规 |
|------|--------|------|------|
| 评估指标 | RankIC / DirAcc / Sharpe | Loss(续写) | Loss(续写) |
| 验证频率 | 每epoch | 每epoch | 每epoch |
| 生成样本 | 无(回测替代) | ✅ 每epoch生成文本 | 无 |

---

## 差异总结

### Kronos 独有

- **两阶段训练**: Tokenizer(自编码) → Predictor(自回归)
- **时序数据**: 滑动窗口 + Z-score归一化
- **回归+分类双模式**: MSE + BCE
- **A/B对比**: Kronos vs LSTM 双架构
- **板块扫描**: sector_scan.py 自动对比

### 医疗/法规共有 (Kronos 缺失)

- **HuggingFace 预训练模型加载**: `AutoModelForCausalLM.from_pretrained()`
- **LoRA 参数高效微调**: PEFT库, 仅训练0.12%参数
- **生成式评估**: 每epoch生成样本文本
- **HuggingFace Trainer 兼容**: Datasets格式

### 医疗项目独有

- **多数据集混合**: MedicalTextDataset + InstructionDataset + MixedDataset
- **ChatML 格式支持**: instruction/response分离
- **Loss masking**: response-only loss

### 法规项目独有

- **简洁性**: 329行, 三项目中最简
- **单一数据源**: 仅RegulationDataset
- **无指令微调**: 纯续写模式
