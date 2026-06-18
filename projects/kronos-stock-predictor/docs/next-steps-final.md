# 下一步方案

> 基于 RankIC=+0.569 (波动率) 最新突破 | 2026-06-17

---

## 当前最优

| 配置 | 方向预测 | 波动率预测 |
|------|---------|----------|
| 数据 | 68 半导体 + 7 长历史 | 68 半导体 |
| 模型 | LSTM (128h, 2层) | LSTM (128h, 2层) |
| 参数 | lookback=180, pred_len=10 | lookback=180, pred_len=10 |
| RankIC | **+0.205** | **🔥 +0.569** |
| 训练 | 3min | 2min |

> **波动率比方向更可预测 (2.8×)**。板块集中 × 长历史 × 简单模型 = 最优。

---

## 下一步行动 (更新)

| # | 行动 | 思路 | 预期 | 状态 |
|---|------|------|------|------|
| 1 🔥 | **波动率→实盘信号** | RankIC=+0.569 → 可交易波动率策略 | 夏普 > 1.5 | ⏳ |
| 2 | **扩展半导体宇宙** | Tushare 全 A 股半导体 (150-200只) | RankIC +0.02 | ⏳ |
| 3 | **生产化部署** | LSTM API + 每日滚动预测 | 可用服务 | ✅ |
| 4 | **时序数据增强** | Jittering/Scaling/TimeWarp | RankIC +0.01 | ⏳ |
| 5 | **集成多 lookback** | 90/180/360 三模型投票 | 降方差 | ⏳ |
| 6 | **方向+波动率融合** | 双信号加权选股 | 夏普提升 | ⏳ |

## 已完成

| 行动 | 结果 |
|------|------|
| Kronos 全流水线 | Tokenizer + Predictor 可训练推理 |
| LSTM A/B 实验 | 方向+波动率双预测 |
| Tushare CSI300 | 329只, 81.7万行 |
| 半导体全量下载 | 194只 |
| 生产 Predictor API | `inference/production.py` |
| 每日管线 | `scripts/daily_pipeline.py` |
| COS 备份 | 模型+数据 |
| 15+ 组实验 | 方向/波动率/跨行业/多模型 |

## 不建议

| 行动 | 理由 |
|------|------|
| 跨行业/CSI300 | 12 次实验均负 RankIC |
| 更复杂模型 | LSTM 已优 BSQ/分类/集成 |
| 更多基本面特征 | PE/PB/PS 未提升 |

## 立即可执行

```bash
# 方案 1: 波动率回测 → 信号生成
python scripts/backtest_volatility.py --data data/processed/semiconductor

# 方案 2: 扩展半导体
python scripts/download_semiconductor_all.py
python scripts/lstm_baseline.py --data data/processed/semiconductor_all --target volatility
```
