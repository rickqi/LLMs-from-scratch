# LLMs-from-scratch 项目群完整说明文档

> 2026-06-18 | 4 个项目 | 1 个学习框架

---

## 项目总览

| # | 项目 | 路径 | 领域 | 模型 | 代码量 | 状态 |
|---|------|------|------|------|--------|------|
| 1 | **Kronos Stock** | `projects/kronos-stock-predictor/` | 金融时序预测 | BSQ+LSTM (PyTorch) | 41py+15md | 🔥 生产就绪 |
| 2 | 医疗文本 | `projects/chinese-medical-text-generation/` | 医学文本生成 | Qwen3+LoRA (HF) | 5py+31md | ✅ 完成 |
| 3 | 法规文本 | `projects/company-regulations-training/` | 法规文本生成 | Qwen3+LoRA (HF) | 4py+4md | ✅ 完成 |
| 4 | **控费ML** | `projects/gbcost-insurance-ml/` | 保险理赔预测 | LightGBM+Optuna | 19py+2md | ✅ 迁移完成 |

---

## 一、Kronos Stock Predictor 🔥

### 定位
半导体 K 线预测模型 — 将金融时序建模转化为离散 token 自回归 + LSTM 直接回归双架构。

### 核心能力

| 能力 | 实现 |
|------|------|
| 波动率预测 | LSTM, RankIC=**+0.579**, 10日窗口 |
| 方向预测 | LSTM, RankIC=+0.205 |
| Kronos架构 | BSQ量化 + Transformer自回归 |
| A/B对比 | Kronos vs LSTM 双模式 |
| 板块扫描 | 电气设备/汽车配件/证券/半导体 |
| 生产部署 | Predictor API + 每日流水线 |
| 数据源 | Tushare (5529只A股) + 本地CSV |

### 架构图

```
数据层: Tushare API / 半导体CSV / CSI300全量
   ↓
模型层: Kronos(BSQ+Transformer) | LSTM(直接回归)
   ↓
评估层: RankIC / DirAcc / Sharpe / Walk-forward回测
   ↓
生产层: Predictor API / daily_pipeline / COS备份
```

### 关键文件

| 文件 | 作用 |
|------|------|
| `model/lstm_model.py` | LSTM 核心模型 |
| `model/kronos_model.py` | Kronos Transformer |
| `model/modules.py` | BSQ量化器 |
| `inference/production.py` | 生产 Predictor API |
| `scripts/daily_pipeline.py` | 每日自动化 |
| `train/train_tokenizer.py` | 两阶段训练 |
| `scripts/download_tushare.py` | Tushare数据下载 |

### 实验历程

```
12组BSQ实验 → RankIC -0.016 (天花板)
   ↓ 切换LSTM
方向预测 → RankIC +0.13
   ↓ 延长历史(2005-2026)
方向预测 → RankIC +0.205
   ↓ 切换目标
波动率预测 → RankIC +0.579 🔥
```

### 生产模型

| 目标 | 模型文件 | RankIC |
|------|---------|--------|
| 波动率 | `outputs/production_vol_model.pt` | +0.579 |
| 方向 | `outputs/production_model.pt` | +0.205 |
| 每日预测 | `outputs/daily/forecast_*.json` | — |

---

## 二、中文医学文本生成

### 定位
基于 Qwen3-0.6B + LoRA 的中文医学诊疗指南文本生成。

### 核心能力

| 能力 | 实现 |
|------|------|
| 续写微调 | 412篇医学文档 → 领域语言适应 |
| 指令微调 | 449条 ChatML QA对 → 问答能力 |
| LoRA高效微调 | 0.12%参数 (0.7M/600M) |
| 数据生成 | DeepSeek V3 API 自动生成QA |
| 断点续传 | checkpoint/training_state.pt |

### 关键文件

| 文件 | 作用 |
|------|------|
| `train_qwen_lora.py` | Qwen3+LoRA 训练(423行) |
| `data_prep.py` | 数据准备 + 样本生成 |
| `generate.py` | 推理 + 交互模式 |
| `scripts/med_qa_generator.py` | QA数据生成(3档) |
| `scripts/cos_backup.py` | COS云备份 |

### 数据流

```
原始MD文档 → data_prep.py → 训练数据
                              ↓
                    Qwen3-0.6B + LoRA 微调
                              ↓
                    续写模型 / 指令模型
```

---

## 三、公司法规文本生成

### 定位
基于 Qwen3-0.6B + LoRA 的公司规章制度文本生成。最简实现(329行训练代码)。

### 核心能力

| 能力 | 实现 |
|------|------|
| 续写微调 | 公司制度文档 → 领域语言 |
| LoRA微调 | 329行最简训练脚本 |
| 断点续传 | checkpoint/training_state.pt |
| COS备份 | 模型 + 数据云备份 |

### 关键文件

| 文件 | 作用 |
|------|------|
| `train_qwen_lora.py` | 训练(329行,三项目最简) |
| `data_prep.py` | 数据准备 |
| `generate.py` | 推理 |
| `scripts/cos_backup.py` | COS备份 |

---

## 四、团险控费 ML 🔥

### 定位
基于 LightGBM + Optuna 的保险理赔金额预测模型。结构化数据ML的最佳实践。

### 核心能力

| 能力 | 实现 |
|------|------|
| 金额预测 | LightGBM + Tweedie损失 |
| 超参优化 | Optuna 贝叶斯优化 |
| Walk-forward回测 | 生产级时序评估 |
| 特征工程 | 71维 → 20+ 特征(Target Encoding等) |
| 数据管道 | polars 高效chunked加载 |
| 配置驱动 | YAML 配置分离 |

### 架构图

```
数据: 13GB CSV (保单+理赔, 71列)
   ↓ polars chunked loading
特征工程: target_encoding + categorical + 过滤
   ↓
模型: LightGBM (Tweedie/Gamma/Regression)
   ↓ Optuna超参优化
评估: Walk-forward回测 + MAE/MAPE/RMSE
   ↓
预测: 理赔金额 + 分位数区间
```

### 关键文件

| 文件 | 作用 |
|------|------|
| `ml/models/amount_predictor.py` | LightGBM 核心模型 |
| `ml/pipeline/train.py` | 训练流水线 |
| `ml/pipeline/tune.py` | Optuna 超参优化 |
| `ml/pipeline/run_backtest.py` | Walk-forward 回测 |
| `ml/data/loader.py` | polars CSV 加载 |
| `ml/data/feature_store.py` | 特征工程 |
| `config_ml.yaml` | ML 配置 |

---

## 横向对比

### 训练方式

| 维度 | Kronos | 医疗 | 法规 | 控费 |
|------|--------|------|------|------|
| 框架 | PyTorch | HF Transformers | HF Transformers | LightGBM |
| 模型类型 | 深度学习 | LLM微调 | LLM微调 | 树模型 |
| 参数量 | 0.5M~4.7M | 0.7M(LoRA) | 0.7M(LoRA) | ~10K |
| 训练时间 | 27s~2h | 30min | 20min | 5min |
| 数据格式 | 时序OHLCV | 文本.md | 文本.md | 结构化CSV |
| 数据量 | 60k~817k行 | 412篇 | ~50篇 | 13GB/百万行 |

### 评估方式

| 项目 | 评估指标 |
|------|---------|
| Kronos | RankIC / DirAcc / Sharpe / Walk-forward |
| 医疗 | Perplexity / 生成质量人工评估 |
| 法规 | Perplexity / 生成质量人工评估 |
| 控费 | MAE / MAPE / RMSE / Walk-forward |

### 生产就绪度

| 项目 | API | 日更 | 备份 | 断点续传 |
|------|-----|------|------|---------|
| Kronos | ✅ Predictor | ✅ daily_pipeline | ✅ COS | ✅ |
| 医疗 | ❌ | ❌ | ✅ COS | ✅ |
| 法规 | ❌ | ❌ | ✅ COS | ✅ |
| 控费 | ✅ predict.py | ❌ | ❌ | ❌ |

---

## 共用模式

### 断点续传 (3/4 项目)

```python
checkpoint_dir = output_dir / "checkpoint"
ckpt_file = checkpoint_dir / "training_state.pt"

# 保存: model + optimizer + scheduler + loss_logs + elapsed
# 恢复: 自动检测, 从断点 epoch 继续
```

Kronos、医疗、法规统一使用此模式。控费尚未实现。

### COS 云备份 (3/4 项目)

```python
# 桶: ins-kq6zz7wo-1313469539 (ap-guangzhou)
# 路径: LLMs-from-scratch/projects/{project}/outputs/
```

### 项目模板

```
project/
├── train_*.py          # 训练脚本
├── data_prep.py        # 数据准备
├── generate.py         # 推理
├── docs/               # 文档
├── scripts/            # 工具脚本
├── outputs/            # 模型产出
├── requirements.txt    # 依赖
└── README.md           # 说明
```

---

## 可互相借鉴

| 来源 → 目标 | 借鉴内容 |
|------------|---------|
| 控费 → Kronos | Optuna超参优化, YAML配置分离 |
| Kronos → 控费 | 断点续传, COS备份, RankIC评估 |
| 医疗 → Kronos | LoRA微调, HuggingFace生态 |
| 医疗 → 控费 | 指令微调(如用LLM做报告) |
| Kronos → 医疗/法规 | 双模型部署, 每日自动化 |
