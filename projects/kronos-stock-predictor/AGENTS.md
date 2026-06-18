# AGENTS.md — Kronos Stock Predictor

> AI Agent 工作流文档。项目结构、当前状态、下一步操作入口。

## 项目概述

A 股 K 线预测，双模型架构：Kronos (BSQ+Transformer) + LSTM (基线/波动率)。

## 当前状态

| 组件 | 状态 | 指标 |
|------|------|------|
| Kronos 模型 | ✅ 完成 | Tokenizer + Predictor 可训练推理 |
| LSTM 模型 | ✅ 完成 | 方向 + 波动率双预测 |
| 波动率预测 | 🔥 突破 | RankIC=+0.569 (2.8×优于方向) |
| 方向预测 | ✅ | RankIC=+0.205 (54长历史LSTM) |
| CSI300 数据 | ✅ | 329只, 81.7万行 |
| 半导体数据 | ✅ | 194只 |
| 生产部署 | ✅ | Predictor API + 每日管线 |
| COS 备份 | ✅ | 模型+数据已备份 |

## 关键文件索引

| 文件 | 用途 | 状态 |
|------|------|------|
| `model/lstm_model.py` | LSTM 方向+波动率模型 | 🔥 主力 |
| `model/kronos_model.py` | Kronos 自回归 Transformer | 实验 |
| `scripts/lstm_baseline.py` | LSTM 快速实验 | 日常 |
| `scripts/lstm_scan.py` | LSTM 参数扫描 | 调优 |
| `scripts/backtest_volatility.py` | 波动率回测 | 评估 |
| `inference/production.py` | 生产预测 API | 部署 |
| `scripts/daily_pipeline.py` | 每日自动预测 | 部署 |

## 操作入口

**快速实验**:
```bash
python scripts/lstm_baseline.py --data data/processed/semiconductor --target direction
python scripts/lstm_baseline.py --data data/processed/semiconductor --target volatility
```

**回测评估**:
```bash
python scripts/backtest_volatility.py --data data/processed/semiconductor --model outputs/lstm_vol.pth
```

**每日预测**:
```bash
python scripts/daily_pipeline.py
```

## 下一步 (按优先级)

| # | 行动 | 思路 | 预期 |
|---|------|------|------|
| 1 | **波动率 → 实盘信号** | 将 RankIC=+0.569 转化为可交易信号 | 夏普 > 1.5 |
| 2 | **扩展半导体** | Tushare 获全板块 150-200 只 | RankIC +0.02 |
| 3 | **数据增强** | Jittering/TimeWarp | 降方差 |
| 4 | **集成模型** | LSTM 方向 + 波动率 + Kronos 投票 | 稳健性 |

## 不建议

- 跨行业/CSI300 (已 12 次实验均负)
- 更复杂模型 (LSTM 已最优)
- 更多基本面特征 (PE/PB 未提升)
