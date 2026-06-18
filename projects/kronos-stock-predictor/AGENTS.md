# AGENTS.md — Kronos Stock Predictor

> v3: 50-epoch volatility breakthrough (RankIC=+0.090)

## 核心指标

| 模型 | 数据 | epochs | 波动率 RankIC | 信号 Sharpe |
|------|------|--------|-------------|-----------|
| v1 | 68只/6年 | 30 | +0.569 | 1.48 |
| v2-30ep | 152只/7年 | 30 | +0.023 | 2.28 |
| **v2-50ep** | **152只/7年** | **50** | **+0.090** | **2.35** |

## 关键发现

- 50 epochs: 波动率 RankIC +0.090 (v2最佳)
- 最优收敛点: epoch 30-35 (之后val loss上升)
- 方向模型过拟合: RankIC 0.086→0.062
- Walk-forward 均值: -0.021 (5ep: -0.251, 12×改善)

## 下一步

### P1: 分行业扫描
```
目标: 找到波动率信号最强的行业
板块: 半导体/医药/消费/新能源/金融
方法: 每个板块独立 LSTM → Walk-forward 对比
预期: 2-3个正 RankIC 板块
```

### P1: 周度 Walk-forward
```
目标: 增加观测点 6→78, 提升统计效力
方法: 每周滚动预测 (vs 每季)
预期: 78数据点 → p<0.01 可期
```

## 快速命令

```bash
# 训练
python scripts/run_retrain_pipeline.py

# 信号回测
python scripts/optimize_signal.py

# Walk-forward
python scripts/run_walkforward.py --epochs 30

# 每日跟踪
python scripts/track_signal.py --cron
```
