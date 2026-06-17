# Tushare 数据获取方案分析

> 2026-06-17

---

## 一、Tushare SDK 状态

| 项目 | 状态 |
|------|------|
| SDK 版本 | 1.4.29 ✅ 已安装 |
| Token | ❌ 未配置 (需 `export TUSHARE_TOKEN=xxx`) |
| API 限速 | 免费 200次/分钟, 付费 500次/分钟 |
| 重试机制 | investment_data 已有 3次重试+1s延迟 |

## 二、investment_data 项目分析

```
/mnt/d/codes/stock/investment_data/
├── tushare/
│   ├── dump_a_stock_eod_price.py     ← 全量下载脚本 (参考实现)
│   ├── update_a_stock_eod_price_to_latest.py ← 增量更新
│   ├── dump_index_weight.py          ← 指数权重
│   └── dump_tushare_stock_list.py    ← 股票列表
├── incremental_update.py             ← 增量QLIB更新 (含限速+重试)
└── daily_update.sh                   ← 每日自动更新
```

### 关键发现

| 项目 | 结论 |
|------|------|
| **CSV 缓存** | ❌ `astock_daily/` 目录为空 — 无历史缓存 |
| **QLIB bin** | ❌ `~/.qlib/qlib_data/cn_data/` 不存在 |
| **代码可用** | ✅ `dump_a_stock_eod_price.py` 可直接参考 |
| **限速处理** | ✅ `incremental_update.py` 已有完整实现 |

### 核心 API (来自 dump_a_stock_eod_price.py)

```python
import tushare as ts
ts.set_token(os.environ["TUSHARE"])
pro = ts.pro_api()

# 获取交易日历 (精确，不浪费调用)
df = pro.trade_cal(exchange='SSE', is_open='1', start_date=..., end_date=...)

# 获取日线 + 复权因子
price_df = pro.daily(trade_date=trade_date)        # OHLCV + pct_chg
adj_df = pro.adj_factor(trade_date=trade_date)      # 复权因子
df = merge(price_df, adj_df, on='ts_code')
df['adj_close'] = df['close'] * df['adj_factor']
```

## 三、数据获取方案

### 方案 A: Tushare 逐日下载 CSI300 ⭐ 推荐

**流程**:
1. 获取交易日历 (1次调用)
2. 逐日下载全市场日线 (每个交易日1次调用)
3. 按品种聚合, 转为 Kronos 格式

**限速策略**:
- 每分钟 ≤180 次 (留 20 次余量)
- 每调用后 sleep(0.35s)
- 失败重试 3 次, 指数退避

**数据量估算**:
- 2015-2026: ~2700 个交易日
- 每次返回 ~5000 只股票
- 全量时间: 2700 / 180 = 15 分钟 (仅限速)
- 实际: ~30-40 分钟 (含网络延迟)

**前提**: 需要 Tushare Token (免费注册 https://tushare.pro)

### 方案 B: Tushare 按股票逐只下载

**流程**:
1. 获取 CSI300 列表
2. 逐只下载日线 `pro.daily(ts_code=...)`

**限速**: 300 只 / 180次/分钟 = ~2 分钟
**劣势**: `daily()` 按股票查询限制更多

### 方案 C: investment_data 的 Dolt 数据库

investment_data 使用 Dolt (Git for data) 存储全市场数据:
- 仓库: `chenditc/investment_data`
- 表: `ts_a_stock_eod_price` (Tushare 来源)
- 但 `dolt clone` 需要 10+ 小时, 不适合

## 四、建议

| 优先级 | 行动 | 前提 |
|--------|------|------|
| **1** | 获取 Tushare Token (免费) | 注册 tushare.pro |
| **2** | 方案 A: 逐日批量下载 CSI300 | Token + ~40min |
| **3** | 转 Kronos 格式 + LSTM 训练 | 数据就绪后 |
