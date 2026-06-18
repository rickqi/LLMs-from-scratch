# AGENTS.md — Kronos Stock Predictor

> AI Agent 工作流。项目结构、当前状态、操作入口。

## 当前状态

| 组件 | 状态 | 关键指标 |
|------|------|---------|
| LSTM 波动率 | ✅ | RankIC=+0.569 |
| LSTM 方向 | ✅ | RankIC=+0.205 |
| 参数优化 | ✅ | top_k=5/20d Sharpe=1.48 |
| 蒙特卡洛 | ✅ | p=0.085 (12点, 效力不足) |
| 数据 v2 | ✅ | 183只, 242K行, 隔离 |
| CPU训练 | ⏳ | 183只方向+波动率待完成 |

## 数据隔离

```
data/processed_real/          # v1 基准 (不可改)
data/semiconductor_v2/        # v2 扩展 (独立)
  raw/183 .pkl + checkpoint.json
  processed/{train,val,test}.pkl
```

## 下一步

| # | 命令 | 说明 |
|---|------|------|
| 🔴 | `nohup python -u scripts/run_retrain_pipeline.py > logs/retrain.log 2>&1 &` | CPU训练 183只 (每epoch ~5min) |
| 🟡 | `python scripts/optimize_signal.py --data test` | 训练完成后参数优化 |
| 🟡 | `python scripts/monte_carlo_backtest.py --n-sims 10000` | 显著性检验 (期望 p<0.05) |

## 关键文件

| 文件 | 用途 |
|------|------|
| `scripts/run_retrain_pipeline.py` | CPU训练管线 |
| `scripts/expand_semiconductor.py` | 数据下载(断点续传) |
| `scripts/optimize_signal.py` | 参数优化 |
| `scripts/monte_carlo_backtest.py` | 蒙特卡洛检验 |
| `model/lstm_model.py` | LSTM模型 |
