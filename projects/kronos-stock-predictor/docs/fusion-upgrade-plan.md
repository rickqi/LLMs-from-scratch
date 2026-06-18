# Kronos → 三路融合升级方案

> 基于 D:\codes\stock 三路融合架构分析 | 2026-06-18

## 一、现状对比

| 维度 | D:\codes\stock (参考) | 我们的 Kronos | 差距 |
|------|---------------------|--------------|------|
| 特征维度 | Alpha158 (158维) | OHLCV (6维) | 26× |
| 模型数量 | 3路融合 | LSTM 单模型 | 3× |
| 股票数 | 21只精选 | 183只全板块 | 互补 |
| 评估方法 | T+1 方向准确率 | Walk-forward RankIC | 互补 |
| 流水线 | daily_pipeline.py | 手动 | ❌ |
| 信号追踪 | accuracy_tracker | track_signal.py | 基本 |

## 二、升级方案

### Phase 1: Qlib 集成 (P0, 2-3h)

```
目标: 183只半导体 + Alpha158 特征 → CatBoost 模型
步骤:
  1. 将 semiconductor_v2 数据转换为 Qlib bin 格式
  2. 运行 Alpha158 特征提取
  3. 训练 CatBoost (分类: 涨/跌)
  4. 评估: 方向准确率 + RankIC
  5. 与 LSTM 信号对比

预期:
  - Alpha158 特征远丰富于 OHLCV 6维
  - CatBoost 在表格数据上优于 LSTM
  - 方向准确率 55-65% (保守估计)
```

### Phase 2: 双路融合 (P0, 1-2h)

```
Qlib CatBoost 方向信号 (权重 0.6)
    +
LSTM 波动率信号      (权重 0.4)
    ↓
融合排名 → 选股

验证: Walk-forward 季度评估，与单模型对比
预期: 融合 > 单独 (多信号互补)
```

### Phase 3: 流水线自动化 (P1, 1-2h)

```
借鉴 daily_pipeline.py 架构:
  Step 1: 数据更新 (Tushare 增量)
  Step 2: Qlib 特征提取 + 预测
  Step 3: LSTM 重训练 + 预测
  Step 4: 双路融合 + 信号输出
  Step 5: 准确率记录 + 告警
```

### Phase 4: LLM Agent 信号 (P2, 可选)

```
借鉴 TradingAgents 12 Agent 辩论:
  - 基本面分析 (财报/新闻)
  - 情绪分析 (社交媒体)
  - 技术分析 (K线形态)
  → AI 评分 (-2 ~ +2)

注: 需要 LLM API 成本，可选
```

## 三、实施路线

```
Week 1:
  □ Qlib bin 格式转换 (30min)
  □ Alpha158 特征提取 (30min)
  □ CatBoost 训练 + 评估 (1h)
  □ LSTM vs CatBoost 对比

Week 2:
  □ 双路融合权重搜索 (1h)
  □ Walk-forward 验证 (1h)
  □ daily_pipeline 框架 (2h)

Week 3+:
  □ 准确率追踪 + 告警
  □ LLM Agent 集成 (可选)
  □ 扩大股票池
```

## 四、预期效果

基于 D:\codes\stock 的实测数据:

| 指标 | 当前 (LSTM only) | 预期 (双路融合) |
|------|-----------------|----------------|
| Walk-forward RankIC | -0.021 | +0.05~0.10 |
| 方向准确率 | ~50% | 55-65% |
| 日均信号数 | 1 (波动率排名) | 2 (方向+波动率) |
| 流水线 | 手动 | 自动化 |
