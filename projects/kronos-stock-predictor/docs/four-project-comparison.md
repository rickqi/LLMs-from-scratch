# 四项目 ML 训练对比分析

> 2026-06-17

---

## 项目概览

| 维度 | Kronos Stock | 医疗文本 | 法规文本 | 团险控费 |
|------|-------------|---------|---------|---------|
| **路径** | `projects/kronos-stock-predictor` | `projects/chinese-medical-text-generation` | `projects/company-regulations-training` | `D:\codes\gbcost-analysys` |
| **领域** | 金融时序预测 | 医学文本生成 | 法规文本生成 | 保险理赔控费 |
| **模型** | Kronos+LSTM (PyTorch) | Qwen3+LoRA (HF) | Qwen3+LoRA (HF) | LightGBM+Optuna |
| **目标** | OHLCV→收益率/波动率 | 文本→续写 | 文本→续写 | 保单特征→理赔金额 |
| **参数量** | 0.5M~4.7M | 0.7M(LoRA) | 0.7M(LoRA) | ~10K(树模型) |
| **训练代码** | 668行 | 423行 | 329行 | ~800行(ml/) |

---

## 训练方式对比

### 模型架构

| 项目 | 架构 | 框架 | 说明 |
|------|------|------|------|
| Kronos | BSQ+Transformer / LSTM | PyTorch | 自研,两阶段训练 |
| 医疗 | Qwen3-0.6B + LoRA | HF Transformers + PEFT | 预训练+微调 |
| 法规 | Qwen3-0.6B + LoRA | HF Transformers + PEFT | 同上 |
| **控费** | **LightGBM** | **lightgbm + Optuna** | **树模型+贝叶斯调参** |

### 数据格式

| 项目 | 输入 | 输出 | 特征数 |
|------|------|------|--------|
| Kronos | OHLCV时序(6维) | 收益率(1维) | 6 |
| 医疗 | 中文文本(.md) | 续写文本 | — |
| 法规 | 中文文本(.md) | 续写文本 | — |
| **控费** | **结构化保单+理赔** | **理赔金额** | **20+** |

### 损失函数

| 项目 | 损失 | 评估 | 
|------|------|------|
| Kronos | MSE(回归) / BCE(分类) | RankIC / DirAcc / Sharpe |
| 医疗 | CrossEntropy(续写) | Perplexity / 生成质量 |
| 法规 | CrossEntropy(续写) | Perplexity / 生成质量 |
| **控费** | **Tweedie/Gamma** | **MAE/MAPE/RMSE** |

### 断点续传

| 特性 | Kronos | 医疗 | 法规 | 控费 |
|------|--------|------|------|------|
| 保存状态 | ✅ model+opt+scheduler+logs | ✅ model+opt+scheduler+logs | ✅ model+opt+scheduler | ❌ 无 |
| 自动恢复 | ✅ | ✅ | ✅ | ❌ |
| 统一格式 | ✅ training_state.pt | ✅ training_state.pt | ✅ training_state.pt | ❌ |

### 超参调优

| 项目 | 方式 | 
|------|------|
| Kronos | 手动扫描 (scripts/lstm_scan.py) |
| 医疗 | 手动指定 |
| 法规 | 手动指定 |
| **控费** | **Optuna 贝叶斯优化** (ml/pipeline/tune.py) |

### 回测/评估

| 项目 | 方式 |
|------|------|
| Kronos | Walk-forward 回测 + RankIC |
| 医疗 | 生成样本人工评估 |
| 法规 | 生成样本人工评估 |
| **控费** | **Walk-forward 回测** (ml/pipeline/run_backtest.py) |

---

## 各自优势

### Kronos (金融时序)
- 两阶段自研训练 (自编码器→自回归)
- A/B 双架构对比 (Kronos vs LSTM)
- 波动率预测突破 (RankIC=+0.579)
- 板块扫描自动对比

### 医疗/法规 (文本生成)
- HuggingFace 生态兼容
- LoRA 参数高效微调 (0.12%)
- 断点续传完善
- 每epoch生成样本评估

### 控费 (保险ML) 🔥
- **LightGBM 树模型**: 结构化数据最优
- **Optuna 超参优化**: 自动化调参
- **Walk-forward 回测**: 生产级评估
- **特征工程**: 20+特征, Target Encoding, 分类处理
- **Tweedie 损失**: 保险理赔金额(零膨胀+长尾)专用
- **数据管道**: polars 高效处理, chunked loading
- **配置驱动**: YAML 配置分离

---

## 可互相借鉴

| 来源 | 借鉴到 | 内容 |
|------|--------|------|
| 控费 → Kronos | **Optuna 超参优化** | 替代手动扫描 |
| 控费 → Kronos | **YAML 配置分离** | config.yaml 替代硬编码 |
| 控费 → Kronos | **polars 数据加载** | chunked loading 大文件 |
| 医疗 → Kronos | **LoRA 微调** | 如果用HF模型 |
| Kronos → 控费 | **断点续传** | training_state.pt 模式 |
| Kronos → 控费 | **RankIC 评估** | 保险场景也可用排序指标 |
