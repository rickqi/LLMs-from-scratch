# Kronos 下一步行动计划

> 基于全部实验结果更新 | 2026-06-16

---

## 实验结果矩阵

| # | 数据 | 品种 | Tokenizer val | Predictor val | RankIC | Win% | 结论 |
|---|------|------|-------------|--------------|--------|------|------|
| 1 | 半导体 | 68 | 0.136 | 4.34 | -0.003 | 53.3 | 基线 |
| 2 | 半导体 P0 | 68 | 0.122 | **3.79** | **+0.016** | 56.7 | ✅ 唯一正 RankIC |
| 3 | 半导体 small | 68 | — | 3.88 | -0.043 | 58.9 | ❌ 过拟合 |
| 4 | +CSI300 | 82 | 0.124 | **3.65** | -0.030 | 68.8 | 最低 val |
| 5 | 全合并 | 96 | **0.117** | 3.77 | -0.039 | 68.8 | 最低 token val |

### 核心洞察

1. **RankIC 瓶颈不在数据量**：68→96 品种，RankIC 不升反降。品种增多≠预测能力提升
2. **mini 是最优规模**：small (4.7M) 在有限数据下过拟合，RankIC 反而转负
3. **半导体板块集中度有价值**：68只半导体的 RankIC 最佳，跨行业混合稀释了板块特有模式
4. **Win Rate 可达 68%**：即使 RankIC 为负，选股胜率仍高于随机

---

## 下一步行动计划

### 🔴 不建议

| 行动 | 理由 |
|------|------|
| 扩大品种池 | 96→更多品种未改善 RankIC，边际收益为零 |
| small/base 模型 | 有限数据下更大模型必然过拟合 |
| akshare 批量下载 | API 不可用，无替代免费方案 |
| 增加 epochs (50+) | 30 epochs 已接近收敛，边际改善有限 |

### 🟢 建议执行

| 优先级 | 行动 | 工作量 | 预期 RankIC | 风险 |
|--------|------|--------|------------|------|
| **P0** | 技术指标特征 (MACD/RSI/布林带/ATR) | 3h | +0.01~0.03 | 低 |
| **P0** | 预测收益率而非价格 | 1h | +0.01~0.02 | 低 |
| **P1** | BSQ 熵正则化 | 4h | +0.01~0.02 | 中 |
| **P1** | 不同预测周期 (3/10/20日) | 1h | 找到最优 | 低 |
| **P2** | Cosine LR + warmup | 0.5h | +0.005 | 低 |
| **P2** | Tushare Pro 全量数据 (需Token) | 2h | +0.02+ | 低 |

### 执行顺序

```
Step 1 (2h): 添加技术指标 → 重新训练 → 评估 RankIC
Step 2 (1h): 预测收益率 → 对比价格预测
Step 3 (1h): 扫描预测周期 3/5/10/20日 → 找最优
Step 4 (4h): BSQ 熵正则化 → 提升 token 质量
```

### P0 立即执行

```bash
# Step 1: 添加技术指标到数据预处理
# 在 preprocessor.py 中添加 MACD/RSI/布林带/ATR

# Step 2: 重新训练 + 回测
python train/train_tokenizer.py --data_dir ./data/processed_ta --output_dir ./outputs/tokenizer_ta --model_size mini --epochs 10 --batch_size 16 --lookback 180
python train/train_predictor.py --tokenizer_path ./outputs/tokenizer_ta/best_model.pt --data_dir ./data/processed_ta --output_dir ./outputs/predictor_ta --model_size mini --epochs 30 --batch_size 16 --lookback 180
```
