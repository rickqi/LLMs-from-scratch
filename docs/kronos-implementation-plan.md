# Kronos 股票 K 线预测模型：完整实施方案

> **基于实际参考项目制定** | 更新日期: 2026-06-15

---

## 项目概述

在 `projects/kronos-stock-predictor/` 下创建完全独立的股票 K 线预测模型项目，与 `projects/chinese-medical-text-generation/` 平行独立。

### 参考项目清单

| 项目 | 路径 | 用于参考 |
|------|------|---------|
| **Kronos 开源模型** | `D:\codes\stock\Kronos\` | 模型架构、BSQ Tokenizer、训练管道 |
| **investment_data** | `D:\codes\stock\investment_data\` | Tushare 数据获取、增量更新、QLIB 格式 |
| **TradingAgents** | `D:\codes\stock\TradingAgents\` | 回测框架、多品种配置、数据服务器抽象 |
| **chinese-medical-text** | `projects/chinese-medical-text-generation/` | 项目模板 (data_prep → train → generate) |
| **LLMs-from-scratch** | `ch04/`, `ch05/` | Transformer 架构、训练循环基础 |

---

## 一、项目目录结构

```
projects/kronos-stock-predictor/
├── README.md                    # 项目说明（快速开始、模型选型）
├── ANALYSIS.md                  # 全面评估（架构分析、风险评估、技术决策）
├── requirements.txt             # Python 依赖
├── .gitignore                   # 排除数据/模型/产出
├── run.sh                       # 一键执行脚本
│
├── config/
│   ├── __init__.py
│   ├── default_config.py        # 全局配置（参考 Kronos finetune/config.py）
│   └── model_configs.py         # 模型规模配置（mini/small/base）
│
├── data/
│   ├── __init__.py
│   ├── downloader.py            # 数据下载：Tushare + akshare（参考 investment_data/tushare/）
│   ├── preprocessor.py          # 数据预处理：Z-score、时间特征、QLIB 格式
│   ├── dataset.py               # PyTorch Dataset：滑动窗口（参考 Kronos finetune/dataset.py）
│   └── symbols.py               # 股票池配置（参考 TradingAgents hs300_tickers.txt）
│   └── data/                    # [gitignore] 原始数据缓存
│   └── processed/               # [gitignore] 预处理后数据
│
├── model/
│   ├── __init__.py
│   ├── kronos_tokenizer.py      # K-line Tokenizer（参考 Kronos model/kronos.py KronosTokenizer）
│   ├── kronos_model.py          # 自回归预测模型（参考 Kronos model/kronos.py Kronos）
│   ├── modules.py               # 基础模块：TransformerBlock, BSQuantizer（参考 Kronos model/module.py）
│   └── predictor.py             # 预测器封装（参考 Kronos model/kronos.py KronosPredictor）
│
├── train/
│   ├── __init__.py
│   ├── train_tokenizer.py       # Tokenizer 训练脚本（参考 Kronos finetune/train_tokenizer.py）
│   ├── train_predictor.py       # 预测器训练脚本（参考 Kronos finetune/train_predictor.py）
│   ├── training_utils.py        # 训练工具函数（DDP、日志、checkpoint）
│   └── losses.py                # 层次化损失函数
│
├── inference/
│   ├── __init__.py
│   ├── generate.py              # 推理脚本（批量/单次/交互模式）
│   └── backtest.py              # 回测框架（参考 Kronos finetune/qlib_test.py）
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # 数据探索
│   ├── 02_tokenizer_training.ipynb   # Tokenizer 训练交互
│   └── 03_prediction_demo.ipynb      # 预测演示
│
├── scripts/
│   ├── download_data.sh         # 数据下载脚本
│   ├── train_all.sh             # 完整训练流程
│   └── evaluate.sh              # 评估脚本
│
├── outputs/                     # [gitignore] 模型权重 + 日志
│   ├── tokenizer/
│   ├── predictor/
│   ├── logs/
│   └── predictions/
│
└── tests/
    ├── test_tokenizer.py
    ├── test_model.py
    └── test_data.py
```

---

## 二、数据管道设计

### 2.1 数据源方案（多源冗余）

```python
# 参考: investment_data/tushare/dump_a_stock_eod_price.py
# 参考: investment_data/incremental_update.py

数据源优先级:
  1. Tushare Pro (付费, 高质量) → ts_a_stock_eod_price
  2. akshare (免费, 全品种) → stock_daily
  3. QLIB cache (本地缓存) → ~/.qlib/qlib_data/cn_data

数据字段（OHLCVA，6维）:
  - open, high, low, close (价格)
  - volume, amount (成交量/额)

字段映射（QLIB normalize → Kronos 格式）:
  adjclose = raw_close * adj_factor
  close/open/high/low → 直接使用
  volume → tushare_vol
  amount → tushare_amount
```

### 2.2 downloader.py 参考实现

```python
# 参考: investment_data/tushare/dump_a_stock_eod_price.py (tushare API)
# 参考: investment_data/incremental_update.py (增量更新逻辑)
# 参考: TradingAgents/tradingagents/default_config.py (多source fallback)

class StockDataDownloader:
    """多源股票数据下载器，支持 Tushare/akshare 双通道"""
    
    def __init__(self, config):
        self.tushare_token = config.tushare_token  # 从环境变量
        self.symbols = config.symbols              # 股票池列表
        self.start_date = config.start_date
        self.end_date = config.end_date
    
    def download_tushare(self, ts_code: str) -> pd.DataFrame:
        """Tushare 通道: 日线 + 复权因子"""
        # 参考 investment_data/tushare/dump_a_stock_eod_price.py:20-27
        pro = ts.pro_api()
        price_df = pro.daily(ts_code=ts_code, start_date=..., end_date=...)
        adj_df = pro.adj_factor(ts_code=ts_code)
        df = pd.merge(price_df, adj_df, on='trade_date')
        df['adjclose'] = df['close'] * df['adj_factor']
        return df[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']]
    
    def download_akshare(self, symbol: str) -> pd.DataFrame:
        """akshare 通道: A股日线"""
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol=symbol, period='daily', ...)
        return df[['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额']]
    
    def incremental_update(self):
        """增量更新：只拉取新交易日"""
        # 参考 investment_data/incremental_update.py:60-72
        last_date = self._read_last_calendar_date()
        trade_dates = pro.trade_cal(exchange='SSE', start_date=last_date, ...)
        for date in trade_dates:
            self._fetch_and_append(date)
```

### 2.3 preprocessor.py 参考实现

```python
# 参考: Kronos/finetune/dataset.py:92-123 (QlibDataset.__getitem__)

def preprocess_series(df: pd.DataFrame, lookback: int, pred_len: int):
    """
    预处理单品种时间序列:
    1. Z-score 归一化（仅基于历史窗口，防止未来信息泄露）
    2. 时间特征提取（minute, hour, weekday, day, month）
    3. 异常值裁剪（clip）
    """
    # 归一化
    past = df.iloc[:lookback]
    mean, std = past[FEATURE_COLS].mean(), past[FEATURE_COLS].std()
    normalized = (df[FEATURE_COLS] - mean) / (std + 1e-5)
    normalized = np.clip(normalized, -5.0, 5.0)  # clip 参考 Kronos config.py:46
    
    # 时间特征（参考 Kronos finetune/dataset.py:60-64）
    df['minute'] = df['datetime'].dt.minute
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    
    return normalized, time_features
```

### 2.4 dataset.py 完整设计

```python
# 参考: Kronos/finetune/dataset.py:9-123 (QlibDataset 完整实现)

class StockDataset(Dataset):
    """
    股票时间序列数据集
    
    设计要点（直接来自 Kronos）:
    - 预计算所有可能的 (symbol, start_index) 对
    - 每个 epoch 随机采样（避免固定顺序导致过拟合）
    - Z-score 归一化严格基于 lookback 窗口
    - 返回 (features, timestamps) 元组
    """
    
    def __init__(self, data_path: str, config: Config, mode: str = 'train'):
        self.config = config
        self.mode = mode
        self.window = config.lookback_window + config.predict_window
        self.feature_list = config.feature_list    # ['open', 'high', 'low', 'close', 'vol', 'amt']
        self.time_feature_list = config.time_feature_list
        
        # 加载预处理数据
        with open(f"{data_path}/{mode}_data.pkl", 'rb') as f:
            self.data = pickle.load(f)
        
        # 预计算索引（参考 Kronos dataset.py:51-71）
        self.indices = []
        for symbol in self.symbols:
            series_len = len(self.data[symbol])
            for i in range(series_len - self.window + 1):
                self.indices.append((symbol, i))
    
    def __getitem__(self, idx):
        # 随机采样（参考 Kronos dataset.py:94-96）
        symbol, start_idx = self.py_rng.choice(self.indices)
        df = self.data[symbol].iloc[start_idx:start_idx + self.window]
        
        # 归一化（仅基于历史窗口）
        past = df.iloc[:self.config.lookback_window]
        mean, std = past[self.feature_list].mean(), past[self.feature_list].std()
        x = (df[self.feature_list] - mean) / (std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)
        
        return (
            torch.FloatTensor(x.values),           # (window, n_features)
            torch.FloatTensor(df[self.time_feature_list].values)  # (window, n_time_features)
        )
```

### 2.5 symbols.py 股票池配置

```python
# 参考: TradingAgents/hs300_tickers.txt, TradingAgents/batch_18stocks.json

# A股核心股票池
CSI300_SYMBOLS = [
    # 金融
    "600036.SH",  # 招商银行
    "601318.SH",  # 中国平安
    # 消费
    "600519.SH",  # 贵州茅台
    "000858.SZ",  # 五粮液
    # 科技
    "002415.SZ",  # 海康威视
    "300750.SZ",  # 宁德时代
    # ... 完整列表参考 hs300_tickers.txt
]

# 指数对标（参考 TradingAgents default_config.py:138-147）
BENCHMARK_MAP = {
    "csi300": "000300.SH",
    "csi500": "000905.SH",
    "csi1000": "000852.SH",
}
```

---

## 三、模型架构设计

### 3.1 核心模块依赖关系

```
modules.py (基础模块)
├── RMSNorm              ← 参考 Kronos model/module.py
├── TransformerBlock     ← 参考 Kronos model/module.py + LLMs ch04/gpt.py
├── BinarySphericalQuantizer ← 参考 Kronos model/module.py:39-99 (完整实现)
├── HierarchicalEmbedding    ← 参考 Kronos model/kronos.py (Kronos.__init__)
├── TemporalEmbedding        ← 参考 Kronos model/kronos.py (Kronos.__init__)
├── DependencyAwareLayer     ← 参考 Kronos model/kronos.py (Kronos.__init__)
└── DualHead                 ← 参考 Kronos model/kronos.py (Kronos.__init__)

kronos_tokenizer.py (Tokenizer)
├── 依赖: modules.TransformerBlock, modules.BinarySphericalQuantizer
└── 参考: Kronos model/kronos.py KronosTokenizer (662行完整实现)

kronos_model.py (自回归模型)
├── 依赖: modules.HierarchicalEmbedding, modules.TemporalEmbedding, ...
└── 参考: Kronos model/kronos.py Kronos (662行完整实现)

predictor.py (预测器)
├── 依赖: kronos_tokenizer.KronosTokenizer, kronos_model.Kronos
└── 参考: Kronos model/kronos.py KronosPredictor (662行完整实现)
```

### 3.2 BSQ 量化器（核心难点，有完整参考）

```python
# 直接参考: Kronos model/module.py:39-99 (BinarySphericalQuantizer)
# 该实现来自官方 BSQ 论文 (https://arxiv.org/pdf/2406.07548.pdf)
# 包含: straight-through estimator、entropy penalty、group quantization

class BinarySphericalQuantizer(nn.Module):
    """
    BSQ 量化器 - 将连续向量二值化为 k-bit 编码
    
    关键参数:
    - embed_dim: 总 bit 数 (如 20)
    - group_size: 分组大小（用于熵计算优化，默认9）
    - beta: commitment loss 权重
    - gamma: entropy penalty 权重
    """
    def quantize(self, z):
        """前向: hard sign {-1,+1}, 反向: STE (z + (zhat - z).detach())"""
        zhat = torch.where(z > 0, torch.tensor(1.), torch.tensor(-1.))
        return z + (zhat - z).detach()
    
    def forward(self, z):
        zq = self.quantize(z)
        q_scale = 1.0 / (self.embed_dim ** 0.5)
        zq = zq * q_scale
        # 计算量化损失...
        return bsq_loss, quantized, z_indices
```

### 3.3 KronosTokenizer 架构

```python
# 直接参考: Kronos model/kronos.py:13-177 (KronosTokenizer 完整实现)

class KronosTokenizer(nn.Module):
    """
    K-line 分词器: Transformer 自编码器 + BSQ 量化
    
    架构流程:
    1. Linear(d_in → d_model)  # 输入投影
    2. Encoder: N×TransformerBlock  # Transformer 编码器
    3. Linear(d_model → codebook_dim)  # 量化前投影
    4. BSQuantizer  # 二值量化 → [s1(粗)|s2(细)]
    5. Decoder: N×TransformerBlock  # Transformer 解码器（双路径）
    6. Linear(d_model → d_in)  # 输出投影
    
    层次化重建:
    - 粗重建: 仅用 s1 bits → L_coarse
    - 细重建: 用完整 [s1|s2] bits → L_fine
    - 量化损失: BSQ commitment loss
    """
    def encode(self, x):   # OHLCV → token indices
    def decode(self, idx): # token indices → OHLCV
    def forward(self, x):  # 训练用: 返回重建+量化损失
```

### 3.4 Kronos Model（自回归预测）

```python
# 直接参考: Kronos model/kronos.py:180-328 (Kronos 完整实现)

class Kronos(nn.Module):
    """
    Decoder-only 自回归预测模型
    
    架构:
    1. HierarchicalEmbedding(s1_ids, s2_ids)  # 层次化 token 嵌入
    2. TemporalEmbedding(stamp)                # 时间特征嵌入
    3. Dropout                                 # token dropout
    4. N×TransformerBlock                      # Transformer 解码器
    5. RMSNorm                                 # 最终归一化
    6. DualHead                                # 双头输出 (s1_logits, s2_logits)
    
    预测流程（参考 kronos.py:239-276 forward）:
    - 前向传播 → s1_logits
    - 从 s1_logits 采样 s1_ids
    - DependencyAwareLayer: 以 s1 嵌入为条件
    - → s2_logits (条件预测)
    
    推理流程（参考 kronos.py:278-328 decode_s1/decode_s2）:
    - decode_s1: 预测 s1 + 返回 context
    - decode_s2: 基于 context + s1_ids 预测 s2
    """
```

### 3.5 模型规模配置

```python
# 参考 Kronos README Model Zoo

MODEL_CONFIGS = {
    "mini": {
        "d_model": 64,      "n_heads": 4,   "n_layers": 4,
        "ff_dim": 256,      "s1_bits": 10,  "s2_bits": 10,
        "context_len": 2048, "params": "4.1M"
    },
    "small": {
        "d_model": 192,     "n_heads": 6,   "n_layers": 8,
        "ff_dim": 768,      "s1_bits": 10,  "s2_bits": 10,
        "context_len": 512,  "params": "28M"
    },
    "base": {
        "d_model": 384,     "n_heads": 12,  "n_layers": 12,
        "ff_dim": 1536,     "s1_bits": 10,  "s2_bits": 10,
        "context_len": 512,  "params": "103M"
    },
}
```

---

## 四、训练管道设计

### 4.1 训练配置

```python
# 参考: Kronos/finetune/config.py (完整 Config 类)

class TrainingConfig:
    # 数据参数（参考 Kronos config.py:11-18）
    lookback_window = 90      # 历史窗口
    predict_window = 10       # 预测窗口
    max_context = 512         # 模型最大上下文
    
    # 训练参数（参考 Kronos config.py:46-68）
    epochs = 30
    batch_size = 50
    tokenizer_lr = 2e-4
    predictor_lr = 4e-5
    adam_beta1 = 0.9
    adam_beta2 = 0.95
    adam_weight_decay = 0.1
    clip = 5.0               # 梯度裁剪 / 数据裁剪
    accumulation_steps = 1
    
    # 数据划分（参考 Kronos config.py:31-38）
    train_range = ["2015-01-01", "2022-12-31"]
    val_range = ["2023-01-01", "2023-12-31"]
    test_range = ["2024-01-01", "2025-06-30"]
```

### 4.2 两阶段训练流程

```bash
# 参考 Kronos finetune 完整 pipeline

# 阶段 1: 训练 Tokenizer（自编码器 + BSQ 量化）
python train/train_tokenizer.py \
    --config config/default_config.py \
    --data_dir data/processed \
    --output_dir outputs/tokenizer

# 阶段 2: 训练 Predictor（自回归预测）
python train/train_predictor.py \
    --tokenizer_path outputs/tokenizer/best_model \
    --data_dir data/processed \
    --output_dir outputs/predictor
```

### 4.3 损失函数

```python
# 参考 Kronos model/kronos.py:239-276 forward 方法中的损失计算逻辑

def tokenizer_loss(x, x_recon_coarse, x_recon_fine, bsq_loss, lambda_quant=1.0):
    """Tokenizer 层次化重建损失"""
    l_coarse = F.mse_loss(x_recon_coarse, x)
    l_fine = F.mse_loss(x_recon_fine, x)
    return l_coarse + l_fine + lambda_quant * bsq_loss

def predictor_loss(s1_logits, s2_logits, s1_targets, s2_targets):
    """Predictor 层次化交叉熵损失"""
    l_s1 = F.cross_entropy(s1_logits.view(-1, vocab_size), s1_targets.view(-1))
    l_s2 = F.cross_entropy(s2_logits.view(-1, vocab_size), s2_targets.view(-1))
    return l_s1 + l_s2
```

---

## 五、推理与回测

### 5.1 KronosPredictor 接口

```python
# 直接参考: Kronos model/kronos.py:482-661 (KronosPredictor 完整实现)

predictor = KronosPredictor(model, tokenizer, max_context=512)

# 单品种预测
pred_df = predictor.predict(
    df=ohlcv_df,
    x_timestamp=hist_timestamps,
    y_timestamp=future_timestamps,
    pred_len=20,
    T=1.0,              # temperature
    top_p=0.9,          # nucleus sampling
    sample_count=5,     # 多轨迹平均
)

# 批量预测（参考 KronosPredictor.predict_batch）
pred_dfs = predictor.predict_batch(df_list, x_ts_list, y_ts_list, ...)
```

### 5.2 回测框架

```python
# 参考: Kronos/finetune/qlib_test.py (回测逻辑)
# 参考: TradingAgents (数据服务器抽象、多品种配置)

def run_backtest(predictor, test_data, config):
    """
    Walk-forward 回测:
    1. 滑动窗口遍历测试集
    2. 对每个时间点：输入历史 90 天 → 预测未来 10 天
    3. 计算预测收益率 → 构建投资组合（Top-K 选股）
    4. 输出累计收益曲线、夏普比率、最大回撤
    """
    for date in test_dates:
        signals = predictor.predict_batch(...)
        portfolio = select_top_k(signals, k=config.n_symbol_hold)
        returns = calculate_returns(portfolio, date)
    
    return BacktestResult(
        cumulative_returns=...,
        sharpe_ratio=...,
        max_drawdown=...,
        benchmark_returns=...  # 参考 TradingAgents benchmark_map
    )
```

---

## 六、分阶段执行计划

### 阶段 1：项目骨架 + 数据管道（第 1-2 周）

| 文件 | 工作量 | 说明 |
|------|--------|------|
| `config/default_config.py` | 0.5天 | 参考 Kronos finetune/config.py |
| `config/model_configs.py` | 0.5天 | 定义 mini/small/base 配置 |
| `data/downloader.py` | 2天 | 参考 investment_data/tushare/*.py |
| `data/preprocessor.py` | 1.5天 | 参考 Kronos finetune/dataset.py |
| `data/dataset.py` | 1.5天 | 参考 Kronos QlibDataset |
| `data/symbols.py` | 0.5天 | 参考 TradingAgents hs300 |
| `data/download_data.sh` | 0.5天 | 一键下载脚本 |
| `.gitignore`, `requirements.txt`, `README.md` | 0.5天 | 项目基础文件 |

**验证标准**：能下载 CSI300 全部成分股日线数据并生成预处理后的 pickle 文件。

### 阶段 2：模型实现（第 2-3 周）

| 文件 | 工作量 | 说明 |
|------|--------|------|
| `model/modules.py` | 3天 | BSQuantizer + TransformerBlock + 其他模块 |
| `model/kronos_tokenizer.py` | 2天 | KronosTokenizer 完整实现 |
| `model/kronos_model.py` | 2天 | Kronos 自回归模型完整实现 |
| `model/predictor.py` | 1天 | KronosPredictor 封装 |
| `tests/test_model.py` | 1天 | 单元测试（张量形状、前向传播） |

**验证标准**：mini 配置下的 Tokenizer + Predictor 能完成前向传播，输出正确形状。

### 阶段 3：训练管道（第 3-4 周）

| 文件 | 工作量 | 说明 |
|------|--------|------|
| `train/losses.py` | 0.5天 | 层次化损失函数 |
| `train/training_utils.py` | 1天 | DDP、日志、checkpoint |
| `train/train_tokenizer.py` | 2天 | 参考 Kronos finetune/train_tokenizer.py |
| `train/train_predictor.py` | 2天 | 参考 Kronos finetune/train_predictor.py |
| `notebooks/02_tokenizer_training.ipynb` | 0.5天 | 训练交互 notebook |

**验证标准**：mini 模型在单品种数据上完成 Tokenizer + Predictor 训练，loss 收敛。

### 阶段 4：推理 + 评估（第 4-5 周）

| 文件 | 工作量 | 说明 |
|------|--------|------|
| `inference/generate.py` | 1天 | 批量/单次/交互推理 |
| `inference/backtest.py` | 2天 | 回测框架 + 可视化 |
| `notebooks/03_prediction_demo.ipynb` | 1天 | 预测演示 |
| `run.sh` | 0.5天 | 一键执行脚本 |
| `ANALYSIS.md` | 1天 | 完整评估文档 |

**验证标准**：完整 pipeline 跑通（数据 → 训练 → 推理 → 回测曲线）。

---

## 七、关键代码复用清单

### 可直接移植的代码（来自 Kronos）

| 源文件 | 目标文件 | 复用内容 | 复用度 |
|--------|---------|---------|--------|
| `Kronos/model/module.py` | `model/modules.py` | BinarySphericalQuantizer (完整的 BSQ 实现) | **90%** |
| `Kronos/model/module.py` | `model/modules.py` | TransformerBlock, RMSNorm, HierarchicalEmbedding | **85%** |
| `Kronos/model/kronos.py` | `model/kronos_tokenizer.py` | KronosTokenizer (encode/decode/forward) | **80%** |
| `Kronos/model/kronos.py` | `model/kronos_model.py` | Kronos (forward/decode_s1/decode_s2) | **80%** |
| `Kronos/model/kronos.py` | `model/predictor.py` | KronosPredictor (predict/predict_batch) | **85%** |
| `Kronos/model/kronos.py` | `inference/generate.py` | auto_regressive_inference, sample_from_logits | **90%** |
| `Kronos/finetune/config.py` | `config/default_config.py` | 配置结构、超参数 | **70%** |
| `Kronos/finetune/dataset.py` | `data/dataset.py` | QlibDataset (滑动窗口+归一化) | **75%** |
| `Kronos/finetune/train_predictor.py` | `train/train_predictor.py` | DDP 训练循环 | **70%** |

### 可直接移植的代码（来自 investment_data）

| 源文件 | 目标文件 | 复用内容 |
|--------|---------|---------|
| `investment_data/tushare/dump_a_stock_eod_price.py` | `data/downloader.py` | Tushare API 调用模式、重试逻辑 |
| `investment_data/incremental_update.py` | `data/downloader.py` | 增量更新逻辑、交易日历获取 |
| `investment_data/incremental_update.py` | `data/preprocessor.py` | tushare → QLIB 数据映射公式 |

### 可直接移植的代码（来自 TradingAgents）

| 源文件 | 目标文件 | 复用内容 |
|--------|---------|---------|
| `TradingAgents/default_config.py` | `config/default_config.py` | 配置管理模式、环境变量覆盖 |
| `TradingAgents/hs300_tickers.txt` | `data/symbols.py` | A股 CSI300 成分股列表 |
| `TradingAgents/default_config.py:138-147` | `inference/backtest.py` | 基准指数映射 |

### 需要从 LLMs-from-scratch 参考的模式

| 源文件 | 用途 |
|--------|------|
| `ch04/gpt.py` GPTModel | TransformerBlock 实现参考（已被 Kronos module.py 替代） |
| `ch05/gpt_train.py` train_model_simple | 训练循环框架（已被 Kronos train_predictor.py 替代） |
| `projects/chinese-medical-text-generation/` | 项目骨架模板（README, run.sh, .gitignore, ANALYSIS.md 格式） |

---

## 八、依赖清单

```txt
# requirements.txt

# 核心框架
torch>=2.2.2
numpy>=1.26
pandas>=2.2.1

# 模型工具
einops>=0.8.1          # 张量操作（Kronos 依赖）
huggingface_hub>=0.34  # 模型上传/下载（可选，用于加载预训练权重）
safetensors>=0.6.2     # 安全模型序列化

# 数据获取
tushare>=1.4.0         # A股数据（需要 token）
akshare>=1.12.0        # 免费数据源（备用）

# 训练
tqdm>=4.66.1           # 进度条

# 可视化
matplotlib>=3.7.1

# 评估
scikit-learn>=1.3.1

# 可选: QLIB 集成（回测用）
# pyqlib
```

---

## 九、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| BSQ 训练不稳定 | 高 | 参考 Kronos 已验证的超参数；EMA 码本更新；先训练 mini 规模 |
| Tushare token 不可用 | 中 | akshare 免费备用通道；使用 Kronos 开源预训练权重跳过自训练 |
| GPU 内存不足 (mini 也需 ~2GB) | 中 | 减小 batch_size；使用 gradient accumulation；CPU fallback for mini |
| 金融数据信噪比低 | 中 | 使用 RankIC 而非 MSE 评估；多轨迹采样降方差 |
| 模型过拟合 | 中 | 时序交叉验证；walk-forward 回测；early stopping |

---

## 十、与现有分析的衔接

本实施方案是对 `docs/kronos-analysis.md` 中可行性评估的延续和细化。analysis.md 提供了：

- 架构逻辑分析（阶段 1-2 的 BSQ + 层次化设计）
- 可行性评估（90% 代码复用度）
- 技术细节补充（BSQ 伪代码、损失函数公式）

本方案在此基础上增加了：

- 完整的项目目录结构和文件清单
- 每个文件对具体参考项目的逐行引用
- 可直接移植的代码清单（9 个文件，75-90% 复用）
- 分周执行计划和验证标准
- 实际依赖清单和风险缓解

建议在开始编码前通读 `docs/kronos-analysis.md` 理解架构原理，然后按本方案的阶段 1→4 逐步执行。
