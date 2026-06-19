# AGENTS.md - Kronos Stock Predictor

> 方法论: VPIN/DPIN 知情交易因子 | RankIC +0.187 (p=0.0145)

## 方法论理由

### 为什么 OHLCV-only 无效

日线 OHLCV (6维: 开/高/低/收/量/额):
  - 一天仅4个价格点 + 2个总量
  - 无法区分"机构建仓"和"散户追涨"
  - 无法区分"恐慌抛售"和"洗盘吸筹"
  - 同根阳线, 买方可能是机构也可能是散户

### 为什么 VPIN/DPIN 有效

VPIN/DPIN 从订单流不平衡中提取知情交易信号:
  - vpin_vol: 上涨日量 vs 下跌日量的不对称 = 方向主导者
  - intraday_rev: 日内反转强度 = 知情者逆势操作的痕迹
  - dpin_stable: 反转稳定性 = 交易行为的持续性

核心思想: 知情交易者会在价格不利于自己时逆势操作,
        这种行为在订单流中留下可检测的不平衡痕迹。

### 学术支撑

PIN (Easley 1996) -> VPIN (Easley 2012) -> DPIN (Chang 2014)
从混合泊松 -> 成交量时钟 -> 日内反转代理

### 实证对比

广发金工 Level-2 真实数据: RankIC 10%, 胜率 74%
我们日线代理数据:       RankIC 最高 0.19, 均值 -0.01
差距: 日线丢失毫秒级逐笔订单信息

## 关键发现

因子工程 > 模型复杂度: 8个VPIN因子带来的提升(+0.30) 远超 LSTM vs CatBoost vs Transformer
