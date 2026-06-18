# Kronos Stock Predictor

基于 **Kronos 架构** 的 A 股 K 线预测模型 — 将金融时间序列建模转化为离散 token 自回归预测。

## 核心思路

```
连续 OHLCV → [Tokenizer: BSQ量化] → 离散 token [s1|s2] → [Decoder-only Transformer] → 未来 K 线
```

## 快速开始

```bash
# 一键运行 (自动生成样本数据 + 训练 + 推理)
bash run.sh

# 指定模型规模
bash run.sh --model small

# 或分步执行
# 1. 数据准备
python -c "from data.preprocessor import generate_sample_data; from config.default_config import Config; generate_sample_data(Config(), './data/processed')"

# 2. 训练 Tokenizer
python train/train_tokenizer.py --data_dir ./data/processed --output_dir ./outputs/tokenizer --model_size mini --epochs 10

# 3. 训练 Predictor
python train/train_predictor.py --tokenizer_path ./outputs/tokenizer/best_model.pt --data_dir ./data/processed --output_dir ./outputs/predictor --model_size mini --epochs 10

# 4. 推理
python inference/generate.py --mode predict --tokenizer_path ./outputs/tokenizer/best_model.pt --predictor_path ./outputs/predictor/best_model.pt --data your_stock.csv
```

## 项目文件

| 文件 | 说明 |
|------|------|
| `run.sh` | 一键执行脚本 |
| `config/default_config.py` | 全局配置（数据、训练超参数、回测参数） |
| `config/model_configs.py` | 模型规模配置（mini/small/base） |
| `data/downloader.py` | 数据下载（Tushare + akshare 双源） |
| `data/preprocessor.py` | 数据预处理（Z-score、时间特征） |
| `data/dataset.py` | PyTorch Dataset（滑动窗口 + 归一化） |
| `data/symbols.py` | A 股沪深300 股票池 |
| `model/modules.py` | 基础模块（BSQ、TransformerBlock 等） |
| `model/kronos_tokenizer.py` | K-line Tokenizer（自编码器 + BSQ） |
| `model/kronos_model.py` | Kronos 自回归预测模型 |
| `model/predictor.py` | KronosPredictor 封装 |
| `train/losses.py` | 层次化损失函数 |
| `train/training_utils.py` | 训练工具（DDP、seed、checkpoint） |
| `train/train_tokenizer.py` | Tokenizer 训练脚本 |
| `train/train_predictor.py` | Predictor 训练脚本 |
| `inference/generate.py` | 推理（单次/批量/交互） |
| `inference/backtest.py` | Walk-forward 回测框架 |
| `tests/test_model.py` | 模型单元测试 |
| `tests/test_data.py` | 数据管道测试 |
| `model/lstm_model.py` | 🔥 LSTM 模型 (方向+波动率) |
| `inference/production.py` | 生产 Predictor API |
| `inference/backtest_enhanced.py` | 增强回测框架 |
| `scripts/` | 14 个实验/部署脚本 |
| `AGENTS.md` | AI Agent 工作流文档 |

## 模型规模

| 模型 | 参数量 | 上下文长度 | 适用场景 |
|------|--------|-----------|---------|
| **mini** | 4.1M | 2048 | 快速原型、CPU 训练 |
| **small** | 28M | 512 | 单品种微调 |
| **base** | 103M | 512 | 多品种预训练 |

## 架构

### 阶段 1: K-line Tokenizer
```
OHLCV(6维) → Encoder(Transformer) → BSQ量化 → [s1(粗, 10bit) | s2(细, 10bit)] → Decoder → 重建OHLCV
```

### 阶段 2: Kronos Predictor
```
离散token序列 → 层次化嵌入 + 时间嵌入 → Decoder-only Transformer → s1预测 → 条件s2预测 → 解码 → 价格
```

## 数据要求

输入 DataFrame 必须包含列: `['open', 'high', 'low', 'close']`
可选: `['volume', 'amount']`

```python
from model.predictor import KronosPredictor

predictor = KronosPredictor(model, tokenizer)
pred_df = predictor.predict(
    df=ohlcv_df,
    x_timestamp=hist_timestamps,
    y_timestamp=future_timestamps,
    pred_len=20,
    T=0.6, top_p=0.9, sample_count=5,
)
```

## 参考

- Kronos 论文: [arXiv:2508.02739](https://arxiv.org/abs/2508.02739)
- Kronos 开源: [github.com/shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
- 投资数据: [github.com/chenditc/investment_data](https://github.com/chenditc/investment_data)
- BSQ 论文: [arXiv:2406.07548](https://arxiv.org/abs/2406.07548)
- LLMs-from-scratch: [github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

## 依赖

```bash
pip install torch numpy pandas einops tushare akshare tqdm matplotlib scikit-learn
```

## 训练结果

> 最终：2026-06-17 | 15+ 组实验完成

| 目标 | 数据 | 模型 | RankIC | 训练 |
|------|------|------|--------|------|
| **🔥 波动率** | **68 半导体** | **LSTM** | **+0.569** | **2min** |
| 方向 | 68 半导体 + 长历史 | LSTM | +0.205 | 3min |
| 方向 | 68 半导体 | LSTM | +0.13 | 27s |
| 方向 | Kronos-small 预训练 | Transformer | -0.016 | 2h |
| 方向 | CSI300 329只 | LSTM | -0.043 | 85s |

> **波动率比方向更可预测 (2.8×)**。板块集中 × 长历史 × 简单模型 = 最优。

### 变更记录

| 日期 | 变更 |
|------|------|
| 2026-06-18 | **扩展半导体**: 183只下载 (242K行), 数据隔离 semiconductor_v2/ |
| 2026-06-18 | **蒙特卡洛检验**: Shuffle/Block/Bootstrap/回撤压力 四重检验 |
| 2026-06-18 | **信号优化**: 24组参数网格搜索, 最优 top_k=5/20d Sharpe=1.48 |
| 2026-06-18 | **CPU训练管线**: run_retrain_pipeline.py (GPU被医学LoRA占用) |
| 2026-06-17 | **🔥 波动率预测突破**: RankIC=+0.569, 2.8×优于方向 |
| 2026-06-17 | **生产部署**: Predictor API + 方向/波动率预测 + 滚动预测 |
| 2026-06-17 | **四方向完成**: 生产/#2长历史/#3日内/#4波动率, COS备份 |
| 2026-06-17 | 全半导体194只 Tushare下载, 54只长历史≥1000天 |
| 2026-06-17 | **长历史突破**: Kronos/data→2005年, RankIC +0.13→+0.205 |
| 2026-06-17 | **Tushare CSI300**: 329 stocks, 817k rows, 2005-2026 |
| 2026-06-17 | **LSTM A/B**: model/lstm_model.py, 三阶段执行方案 |
| 2026-06-16 | Kronos-small预训练 + BSQ熵 + pred_len扫描 |
| 2026-06-15 | 端到端流水线 + 68半导体 + 15 pytest通过 |
| 2026-06-16 | Kronos-small预训练 + CosineLR + BSQ熵 + pred_len扫描 |
| 2026-06-15 | 端到端流水线 + 68半导体 + 15 pytest通过 |
