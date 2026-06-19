# VPIN/DPIN 因子实施方案

> 基于广发金工 Level-2 知情交易因子研究 | 2026-06-18

## 突破

OHLCV 6维 → VPIN/DPIN 14维: RankIC -0.117 → +0.187 (p=0.0145)

## 因子定义

| 因子 | 公式 | 含义 |
|------|------|------|
| `vpin_vol` | Σ(vol×sign(ret)) / Σ|vol| (5日) | 成交量方向失衡 |
| `vpin_vol_20` | 同上 (20日) | 中期失衡 |
| `vpin_amt` | Σ(amt×sign(ret)) / Σ|amt| (5日) | 成交额失衡 |
| `intraday_rev` | (close-open)/(high-low) | 日内反转强度 |
| `intraday_rev_std` | rolling std(10) | 反转波动 |
| `intraday_rev_mean` | rolling mean(10) | 反转均值 |
| `dpin_stable` | mean/std | DPIN 稳定性 |
| `am_pm_ratio` | close/open ratio (5日) | 开盘-收盘比 |

## 验证计划

| 步骤 | 内容 | 预期 |
|------|------|------|
| 1 | 多窗口验证: 2023→2024, 2024→2025 | RankIC 稳定性 |
| 2 | Walk-forward: 季度滚动 | 样本外一致性 |
| 3 | Monte Carlo: 每窗口 10000 次 | 统计显著性 |
| 4 | 信号回测: 选股策略 | Sharpe |
