# Kronos Stock Predictor — 详细技术文档

## 项目定位

半导体 K 线预测系统，覆盖 68 只 A 股半导体股票，实现方向预测 (RankIC=+0.205) 和波动率预测 (RankIC=+0.579)。

## 技术架构

```
数据层                   模型层                    评估层              生产层
Tushare API ─┐
半导体 CSV  ─┤→ BSQ Tokenizer → Kronos Transformer → RankIC ─┐
CSI300全量  ─┤→ LSTM (直接回归)                    → DirAcc  ├→ Predictor API
              │                                     → Sharpe  │→ daily_pipeline
              └→ StockDataset (滑动窗口+Z-score)    → Walk-fwd┘→ COS 备份
```

## 核心模块

| 模块 | 文件 | 行数 | 说明 |
|------|------|------|------|
| BSQ量化器 | `model/modules.py` | 410 | Binary Spherical Quantization |
| Kronos Tokenizer | `model/kronos_tokenizer.py` | 335 | 自编码器+BSQ |
| Kronos Model | `model/kronos_model.py` | 489 | Decoder-only Transformer |
| LSTM Model | `model/lstm_model.py` | 150 | 直接回归LSTM |
| Predictor | `model/predictor.py` | 351 | 推理API |
| 训练-Tokenizer | `train/train_tokenizer.py` | 253 | 两阶段训练Stage1 |
| 训练-Predictor | `train/train_predictor.py` | 290 | 两阶段训练Stage2 |
| 数据处理 | `data/dataset.py` | 130 | 滑动窗口Dataset |
| 生产推理 | `inference/production.py` | 82 | Predictor class |

## 数据管道

```
Tushare API (5529 A股) → download_tushare.py → processed_tushare/
半导体 CSV (69只)      → convert_semiconductor.py → processed_real/
Kronos/data (30只)     → 合并 → processed_csi300/
```

## 实验全记录 (15组)

| # | 实验 | RankIC | 结论 |
|---|------|--------|------|
| 1 | Kronos mini, 68半导, 3ep | -0.003 | 基线 |
| 2 | Kronos mini, 30ep, lookback=180 | -0.016 | BSQ天花板 |
| 3 | Kronos small, 68半导 | -0.043 | 过拟合 |
| 4 | Kronos, 96混合品种 | -0.039 | 品种≠效果 |
| 5 | Kronos, 技术指标19feat | N/A | 恶化 |
| 6 | Kronos-small预训练 | -0.110 | 不适用 |
| 7 | LSTM, 68半导, pl=10 | +0.131 | LSTM突破 |
| 8 | LSTM分类 | -0.008 | 不如回归 |
| 9 | LSTM集成 | +0.007 | 不如单模型 |
| 10 | LSTM, 329 CSI300 | -0.067 | 板块稀释 |
| 11 | LSTM, 68半导+长历史 | +0.205 | 方向最优 |
| 12 | LSTM, 194全半导体 | +0.010 | 新股稀释 |
| 13 | LSTM波动率, 68半导 | +0.579 | 🔥 全局最优 |
| 14 | 电气设备LSTM | +0.133 | 可用 |
| 15 | 汽车配件LSTM | +0.125 | 可用 |

## 依赖

```
torch, numpy, pandas, tushare, scipy, scikit-learn,
matplotlib, einops, huggingface_hub, safetensors
```

## 运行方式

```bash
# 每日自动化预测
python scripts/daily_pipeline.py

# 训练新模型
python train/train_tokenizer.py --data_dir ./data/processed_real --model_size mini
python train/train_predictor.py --tokenizer_path ./outputs/tokenizer/best_model.pt --model_size mini

# 推理
python -c "from inference.production import Predictor; ..."
```
