# Kronos 下一步执行方案

> 基于 10 组实验结果 | 2026-06-17

---

## 当前最优模型

| 配置 | 详情 |
|------|------|
| 模型 | LSTM (128 hidden, 2 layers) |
| 数据 | 68 半导体, 2018-2026 |
| 参数 | lookback=180, pred_len=10 |
| RankIC | **+0.131** |
| 训练时间 | 27s |
| 代码 | `model/lstm_model.py` |

---

## 执行计划

### Phase 1: 生产化（本周）

| 步骤 | 命令 | 产出 |
|------|------|------|
| 1.1 | 训练最终 LSTM 模型 (epochs=50) | `outputs/lstm_final.pt` |
| 1.2 | 封装预测 API | `inference/lstm_predict.py` |
| 1.3 | 生成 30 日滚动预测报告 | `outputs/rolling_forecast.json` |

```bash
# 1.1 最终训练
python -c "
from model.lstm_model import LSTMModel, StockDataset, train_lstm
import pickle, torch
data = {m: pickle.load(open(f'data/processed_real/{m}_data.pkl','rb')) for m in ['train','val']}
model = train_lstm(data['train'], data['val'], epochs=50, lr=5e-4)
torch.save(model.state_dict(), 'outputs/lstm_final.pt')
"
```

### Phase 2: 特征增强（下周）

| 步骤 | 思路 | 预期 |
|------|------|------|
| 2.1 | 添加板块指数 (费城半导体 SOX) | 宏观背景 |
| 2.2 | 涨跌停/振幅/换手率 | 已有数据未使用 |
| 2.3 | 滞后 1/2/3 日特征 | 时序自相关 |

### Phase 3: 多频率（本月）

| 步骤 | 前提 | 预期 |
|------|------|------|
| 3.1 | Tushare 获取 30min K 线 | 需 Token (已有) |
| 3.2 | LSTM 日内模型 | 更高信噪比 |

---

## 不建议

| 行动 | 理由 |
|------|------|
| 继续 Kronos/BSQ 优化 | LSTM 已证明 BSQ 是瓶颈 |
| 扩大品种池 | CSI300 验证板块集中度 > 数据量 |
| 更复杂模型 | 简单 LSTM 已最优 |

---

## 即时执行

```bash
# Phase 1.1: 训练最终 LSTM 模型 (5 分钟)
cd projects/kronos-stock-predictor
python3 -c "
from model.lstm_model import LSTMModel, StockDataset, train_lstm
import pickle, torch
data = {
    'train': pickle.load(open('data/processed_real/train_data.pkl','rb')),
    'val': pickle.load(open('data/processed_real/val_data.pkl','rb')),
    'test': pickle.load(open('data/processed_real/test_data.pkl','rb')),
}
model = train_lstm(data['train'], data['val'], epochs=50, lr=5e-4, device='cuda:0')
torch.save(model.state_dict(), 'outputs/lstm_final.pt')
print('LSTM final model saved')
"
```
