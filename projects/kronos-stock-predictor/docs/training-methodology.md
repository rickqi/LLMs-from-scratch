# Kronos Stock Predictor — 当前训练方法完整说明

> 2026-06-18 | 最新: VPIN/DPIN 因子 + CustomLSTM

---

## 一、数据管线

### 1.1 数据源

```
Tushare Pro API (daily接口)
  ↓ pro.daily(ts_code, start_date, end_date)
单只股票日线: trade_date, open, high, low, close, vol, amount
  ↓ 断点续传 (checkpoint.json)
semiconductor_v2/raw/{symbol}.pkl  (183只, 242K行)
```

### 1.2 数据划分

当前最优分法 (适配 2025-2026 体制切换):

```python
train_end = "2025-09-30"   # 2025年前9个月
test_start = "2026-01-01"  # 2026年至今

# 过滤条件
train: len(df) >= 60天   # 最少60天训练数据
test:  len(df) >= 10天   # 最少10天测试数据
```

### 1.3 特征计算

原始 6 维 OHLCV + 8 维 VPIN/DPIN 代理因子:

```python
# OHLCV (6维) - 原始日线数据
FEATURES = ["open", "high", "low", "close", "vol", "amt"]

# VPIN/DPIN 代理因子 (8维) - 从日线中提取
df['vpin_vol']     = (vol * sign(ret)).rolling(5).sum() / abs(vol).rolling(5).sum()
df['vpin_vol_20']  = 同上, 20日窗口
df['vpin_amt']     = (amt * sign(ret)).rolling(5).sum() / abs(amt).rolling(5).sum()
df['intraday_rev']       = (close - open) / (high - low + 1e-5)
df['intraday_rev_std']   = intraday_rev.rolling(10).std()
df['intraday_rev_mean']  = intraday_rev.rolling(10).mean()
df['dpin_stable']        = mean / (std + 1e-5)
df['am_pm_ratio']        = (close / open).rolling(5).mean()

# 最终: 14维输入
VPIN_FEATURES = FEATURES + [vpin_vol, vpin_vol_20, vpin_amt,
                             intraday_rev, intraday_rev_std, intraday_rev_mean,
                             dpin_stable, am_pm_ratio]
```

---

## 二、模型架构

### 2.1 CustomLSTM

```python
class CustomLSTM(nn.Module):
    """支持可变输入维度的双层LSTM"""
    def __init__(self, input_dim=14, hidden=128, num_layers=2):
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers,
            batch_first=True, dropout=0.2
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),  # 128 → 64
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)                 # 64 → 1
        )
    
    def forward(self, x):
        # x: (batch, lookback, input_dim)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # 取最后时间步
```

| 参数 | 值 | 说明 |
|------|-----|------|
| input_dim | 14 | OHLCV(6) + VPIN/DPIN(8) |
| hidden | 128 | LSTM隐藏维度 |
| num_layers | 2 | LSTM层数 |
| dropout | 0.2 | 防止过拟合 |
| 总参数量 | ~200K | 轻量级 |

### 2.2 数据集

```python
class StockDS(Dataset):
    """滑动窗口采样"""
    def __init__(self, data, features, lookback=60, pred_len=3, max_samples=20000):
        # 遍历所有股票的所有窗口
        for sym, df in data.items():
            vals = df[features].values.astype(np.float32)
            for i in range(len(vals) - lookback - pred_len):
                x = vals[i : i+lookback]
                # Z-score归一化 (每个窗口独立)
                x_norm = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-5)
                # 标签: 未来pred_len天的收益率
                y = (vals[i+lookback+pred_len-1, 3] - vals[i+lookback-1, 3]) \
                    / (vals[i+lookback-1, 3] + 1e-5)
                if abs(y) < 0.2:  # 过滤极端值
                    self.samples.append((x_norm, y))
        self.n = min(len(self.samples), max_samples)
```

### 2.3 训练循环

```python
# 超参数
LOOKBACK = 60      # 回溯窗口(交易日)
PRED_LEN = 3       # 预测未来N天
EPOCHS = 30        # 训练轮数
BATCH_SIZE = 128   # 批次大小
LR = 1e-3          # 学习率

model = CustomLSTM(input_dim=14, hidden=128).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

for ep in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()
    
    # 验证 + 保存最佳模型
    model.eval()
    val_loss = evaluate(model, val_loader)
    if val_loss < best_val:
        best_val = val_loss
        best_state = model.state_dict()

model.load_state_dict(best_state)
```

---

## 三、评估体系

### 3.1 预测评估

```python
# 对测试集每只股票做一次预测
for sym, df in test_data.items():
    vals = df[features].values.astype(np.float32)[-LOOKBACK:]
    x_norm = (vals - vals.mean(0)) / (vals.std(0) + 1e-5)
    pred = model(torch.FloatTensor(x_norm).unsqueeze(0).to(DEVICE)).item()
    
    # 实际波动率
    rets = df.close.pct_change().dropna()
    actual = rets.std()
    
    preds.append(pred)
    actuals.append(actual)

# RankIC: 预测排序 vs 实际排序的相关性
rankic, p_value = spearmanr(preds, actuals)
```

### 3.2 统计检验

```python
# Monte Carlo: 10,000次随机打乱预测值
for _ in range(10000):
    np.random.shuffle(preds)
    ic = spearmanr(preds, actuals)[0]
    shuffled_ics.append(ic)

p_value = 1 - percentileofscore(shuffled_ics, real_ic) / 100
significant = p_value < 0.05
```

### 3.3 信号回测

```python
# 波动率选股策略
# 每rebalance天: 选预测波动率最低的top_k只 → 等权买入
for date in rebalance_dates:
    for sym in symbols:
        pred_vol = predict(model, data[sym].loc[:date])
    selected = sorted(preds, key=vol)[:top_k]
    # 等权买入, 记录权益
    equity = cash + sum(holdings * prices)

sharpe = sqrt(252/rebalance) * returns.mean() / returns.std()
```

---

## 四、当前结果

| 指标 | 值 |
|------|-----|
| 最佳单窗 RankIC | +0.187 (2025→2026, VPIN/DPIN) |
| 多窗均值 | -0.012 (4窗口, 3/4正) |
| 蒙特卡洛 p-value | 0.0145 (单窗显著) |
| 方向准确率 | ~53% |
| 训练时间 | ~4min (GPU, 30epochs) |

---

## 五、文件清单

| 文件 | 功能 |
|------|------|
| `scripts/expand_semiconductor.py` | 数据下载 (断点续传) |
| `scripts/train_catboost.py` | VPIN/DPIN因子计算 + 训练 |
| `scripts/optimize_signal.py` | 参数网格搜索 |
| `scripts/monte_carlo_backtest.py` | 统计显著性检验 |
| `scripts/sector_scan.py` | 行业板块扫描 |
| `scripts/run_retrain_pipeline.py` | 完整训练管线 |
| `scripts/track_signal.py` | 每日信号跟踪 |
| `scripts/eval_accumulated.py` | 累积信号回看 (D任务) |
| `model/lstm_model.py` | LSTM 模型定义 |
| `docs/vpin-factor-plan.md` | VPIN因子实施方案 |
| `docs/final-summary-report.md` | 项目总结报告 |
