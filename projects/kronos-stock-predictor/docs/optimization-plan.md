# Kronos Stock Predictor — 完整优化方案

> 基于 GitHub OSS 对比分析 | 2026-06-21

---

## 一、现状诊断

### 1.1 当前架构

```
Stage 1: K-line Tokenizer (Transformer AE + BSQ 量化)
         OHLCV(6维) → 层次化离散 token (s1粗/s2细, 各10bit)

Stage 2: Decoder-only Transformer 自回归预测
         规模: mini(4M) / small(28M) / base(102M)
         上下文: 512-2048 tokens

因子层: VPIN/DPIN 知情交易因子 (8维)
         RankIC +0.187 (p=0.0145)

优化层: GRPO — Sharpe Ratio 直接优化 (LSTM only, 未接入 Transformer)

回测:   自建 backtest_enhanced.py
         评估: RankIC, Sharpe, MaxDD, WinRate
```

### 1.2 与 GitHub OSS 差距

| 对比维度 | Kronos 现状 | MASTER (AAAI 2024) | FinCast (CIKM 2025) | Meridian.AI | neotema/Kronos |
|---------|------------|-------------------|---------------------|-------------|----------------|
| **回测框架** | 自建 | **Qlib 集成** (AR/IR/IC) | 自建 | 自建 | 自建 |
| **评估指标** | RankIC+Sharpe+MaxDD | AR/IR/IC/ICIR/回撤 | 多基准 MAE/sMAPE | 涨跌准确率 | 多基准 |
| **模型多样性** | Transformer + LSTM | Transformer only | **MoE** (多专家) | **Mamba-2 + GQA** | Transformer only |
| **数据覆盖** | CSI 300 + 半导体 | CSI 300/500/800 | **20B 金融点** | 全球任意股票 | **45+ 交易所** |
| **预训练** | 无 | 无 | **20B token 预训练** | 2h 自动重训 | **Foundation Model** |
| **因子体系** | VPIN/DPIN (8维) | 市场引导 | MoE 领域路由 | 44 技术指标 | 纯 K-line token |
| **发布形式** | 本地 | ⭐500+ | ⭐124 | HF 模型 | HF 模型 |

### 1.3 核心瓶颈

1. **回测不可比**: 自建回测 vs Qlib 标准化，无法与学术基准对齐
2. **模型未规模化**: mini(4M) 太小，未训练 small/base 配置
3. **因子利用率低**: VPIN/DPIN 仅用于 LSTM 信号，未输入 Transformer
4. **GRPO 仅 LSTM**: 强化学习优化未接入主模型
5. **无预训练**: 每个实验从零训练，浪费计算
6. **单资产**: 仅 CSI 300 + 半导体，缺乏跨市场泛化

---

## 二、优化方案（分阶段）

### Phase 1: Qlib 回测集成（P0 — 基础建设）

**目标**: 使 Kronos 回测结果与学术界/业界可比

**为什么选 Qlib**:
- MASTER (AAAI 2024) 项目官方推荐 Qlib 实现
- 提供 AR (Annual Return)、IR (Information Ratio)、IC (Information Coefficient)、ICIR 等标准指标
- 内置 TopkDropout、WeightDecay 等投资组合构建器
- 中国 A 股数据源 (CSI 300/500/800) 开箱即用

**实施步骤**:

```bash
# 1. 安装 Qlib
pip install pyqlib

# 2. 下载 A 股数据
python -m qlib.run.get_data --region cn --interval 1d

# 3. 创建 Kronos Qlib 模型包装器
# 新建: model/qlib_kronos.py
```

```python
# model/qlib_kronos.py — Qlib 模型接口
import qlib
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.contrib.model.pytorch_utils import prepare_seqlabel_data

class KronosQlibModel(Model):
    """将 Kronos Predictor 包装为 Qlib 兼容模型"""
    
    def __init__(self, predictor_path, tokenizer_path, **kwargs):
        self.predictor = KronosPredictor.load(predictor_path)
        self.tokenizer = KronosTokenizer.load(tokenizer_path)
    
    def fit(self, dataset: DatasetH):
        # 使用预训练模型，跳过 fit
        pass
    
    def predict(self, dataset: DatasetH) -> np.ndarray:
        # Qlib dataset → Kronos 格式 → 预测
        df = dataset.prepare("train")
        predictions = self.predictor.predict_batch(df)
        return predictions
```

```yaml
# config/qlib_config.yaml
qlib:
  market: &market csi300
  benchmark: &benchmark SH000300
  
  data_handler:
    class: Alpha158
    module_path: qlib.contrib.data.handler
    kwargs:
      start_time: "2018-01-01"
      end_time: "2024-12-31"
      instruments: *market
  
  model:
    class: KronosQlibModel
    kwargs:
      predictor_path: "./outputs/predictor/best_model.pt"
      tokenizer_path: "./outputs/tokenizer/best_model.pt"
  
  backtest:
    start_time: "2022-01-01"
    end_time: "2024-12-31"
    account: 100000000
    benchmark: *benchmark
    exchange_kwargs:
      limit_threshold: 0.095
      deal_price: close
```

**预期产出**:
- `outputs/qlib_results/` — 标准化回测报告
- 指标: AR, IR, MaxDD, IC, ICIR, RankIC, 换手率
- 可与 MASTER 论文结果直接对比

**投入**: ~4h (安装 + 包装器 + 配置 + 验证)

---

### Phase 2: 模型规模化 + 因子融合（P0 — 性能提升）

**目标**: 训练更大模型 + 将 VPIN/DPIN 因子输入 Transformer

**2.1 模型规模升级**

当前只训练了 mini(4M)。应依次训练：

| 配置 | 参数量 | 上下文 | 建议数据量 | 预计训练时间 (RTX5080) |
|------|--------|--------|-----------|----------------------|
| mini | 4.1M | 2048 | ~100 只股票 | 2h |
| small | 24.7M | 512 | ~300 只股票 | 6h |
| base | 102.3M | 512 | CSI 全市场 | 24h+ (需多 GPU) |

**2.2 VPIN/DPIN 因子注入 Transformer**

当前架构仅使用 OHLCV(6维) 作为输入。应将 VPIN/DPIN 因子作为额外特征：

```python
# 修改 model/kronos_tokenizer.py
class KronosTokenizer:
    def __init__(self, ..., use_factor_features=True):
        # 原始 OHLCV: d_in=6
        # + VPIN/DPIN 因子: d_in=14 (6+8)
        if use_factor_features:
            self.d_in = 14  # OHLCV + 8 因子
        
    def forward(self, ohlcv, factors=None):
        x = ohlcv  # (B, T, 6)
        if factors is not None:
            x = torch.cat([x, factors], dim=-1)  # (B, T, 14)
        return self.encode(x)
```

**预期收益**:
- VPIN/DPIN 因子 RankIC 已验证 +0.187
- 注入 Transformer 后预期 RankIC 提升至 0.25-0.30
- 模型容量提升: 4M → 28M → 性能跃升

**投入**: ~8h (训练 small) + ~2h (因子融合代码)

---

### Phase 3: MoE 多专家架构（P1 — 架构创新）

**目标**: 引入 Mixture-of-Experts 处理多资产类型

**参考**: FinCast 的 MoE 设计

```python
# model/moe_kronos.py
class MoEKronos(Kronos):
    """MoE-enhanced Kronos with domain-specific experts"""
    
    def __init__(self, num_experts=4, expert_capacity=2, **kwargs):
        super().__init__(**kwargs)
        
        # 领域路由器
        self.router = nn.Sequential(
            nn.Linear(kwargs['d_model'], num_experts),
            nn.Softmax(dim=-1)
        )
        
        # 领域专家 (各行业不同的 Transformer Block)
        self.experts = nn.ModuleList([
            TransformerBlock(kwargs['d_model'], kwargs['n_heads']) 
            for _ in range(num_experts)
        ])
        
        # 专家标签 (基于股票所属行业)
        # 0: 金融, 1: 科技, 2: 消费, 3: 制造
        self.register_buffer('sector_map', torch.zeros(1000, dtype=torch.long))
    
    def forward(self, x, sector_ids=None):
        # Top-2 expert routing
        router_weights = self.router(x.mean(dim=1))  # (B, num_experts)
        top2_weights, top2_indices = torch.topk(router_weights, 2, dim=-1)
        
        # Sparse MoE forward
        output = torch.zeros_like(x)
        for i in range(2):
            expert_idx = top2_indices[:, i]
            weight = top2_weights[:, i].unsqueeze(-1).unsqueeze(-1)
            for j in range(len(self.experts)):
                mask = (expert_idx == j)
                if mask.any():
                    output[mask] += weight[mask] * self.experts[j](x[mask])
        
        return output
```

**预期收益**:
- 行业特化提升预测精度
- 稀疏计算保持推理效率
- 跨行业泛化能力增强

**投入**: ~10h (实现 + 训练 + 评测)

---

### Phase 4: 强化学习管线升级（P1 — GRPO → RLHF）

**目标**: 将 GRPO 从 LSTM 扩展到 Transformer，建立完整 RL 管线

**当前**: GRPO 仅优化 LSTM，且为单步 Sharp 优化
**目标**: 多轮 RL (GRPO → PPO → TRPO)，优化 Transformer 预测

```python
# scripts/train_kronos_rl.py
"""
Kronos RL 训练流程:
  1. 预训练 Transformer (Stage 2, MSE loss)
  2. GRPO: Sharpe 直接优化
  3. PPO: 带 clipping 的稳定策略优化
  4. 可选 TRPO: 信任域约束更保守
"""

class KronosRL:
    def __init__(self, model, ref_model, beta=0.1):
        self.model = model          # 策略模型
        self.ref_model = ref_model  # 参考模型 (冻结)
        self.beta = beta            # KL 惩罚系数
    
    def grpo_step(self, batch):
        """GRPO: Group Relative Policy Optimization"""
        group_size = 8
        predictions = []
        for _ in range(group_size):
            pred = self.model.sample(batch)  # 随机采样
            predictions.append(pred)
        
        # 计算相对优势 = 个体 Sharpe - 群体平均 Sharpe
        sharpes = [self.compute_sharpe(p, batch.labels) for p in predictions]
        mean_sharpe = np.mean(sharpes)
        advantages = [s - mean_sharpe for s in sharpes]
        
        # 策略梯度更新
        loss = -np.mean([a * self.model.log_prob(p, batch) 
                        for a, p in zip(advantages, predictions)])
        return loss
    
    def ppo_step(self, batch):
        """PPO: Proximal Policy Optimization"""
        old_log_prob = self.model.log_prob(batch).detach()
        new_log_prob = self.model.log_prob(batch)
        ratio = torch.exp(new_log_prob - old_log_prob)
        
        advantage = self.compute_advantage(batch)
        
        # Clipped surrogate objective
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
        loss = -torch.min(surr1, surr2).mean()
        
        # KL 惩罚 (防止偏离参考模型)
        kl_penalty = self.compute_kl(self.model, self.ref_model, batch)
        return loss + self.beta * kl_penalty
```

**预期收益**:
- Sharpe 从 1.2 提升到 1.8-2.0 (参考 GRPO 实验已有 +56%)
- RL 训练使模型从"预测准确"转为"交易盈利"
- 直接对齐金融市场目标函数

**投入**: ~12h (RL 管线实现 + 训练 + 评测)

---

### Phase 5: 多资产 + 预训练（P2 — 规模化）

**目标**: 扩展数据覆盖 + 预训练 Foundation Model

**5.1 数据扩展**

```
当前: CSI 300 + 183 半导体
目标: CSI 全市场 (300+500+800) + 港股通 + 美股中概
```

**5.2 预训练策略 (参考 FinCast 20B 金融点预训练)**

```bash
# Stage 0: 大规模预训练 (自监督, 无需标注)
python train/train_kronos_pretrain.py \
    --data_dir ./data/all_markets \
    --model_size base \
    --context_len 2048 \
    --batch_size 64 \
    --epochs 10 \
    --output_dir ./outputs/pretrained_base

# Stage 1: Tokenizer (使用预训练权重初始化)
python train/train_tokenizer.py \
    --pretrained ./outputs/pretrained_base \
    --data_dir ./data/csi300

# Stage 2: Predictor 微调 (使用预训练权重初始化)
python train/train_predictor.py \
    --pretrained_tokenizer ./outputs/tokenizer \
    --pretrained_predictor ./outputs/pretrained_base \
    --data_dir ./data/csi300
```

**预期收益**:
- 预训练后微调 RankIC +0.05-0.10
- 跨市场零样本能力 (预训练见过港股 → 微调 A 股时泛化更好)
- 模型鲁棒性提升

**投入**: ~40h (数据采集 + 预训练 + 验证)

---

## 三、实施路线图

```
Week 1-2 (P0):   Qlib 集成 + 因子融合 ✅ 已完成
  ├── scripts/qlib_metrics.py — Qlib 兼容指标引擎
  ├── data/factors.py — VPIN/DPIN 8 因子计算
  └── config/model_configs.py — d_in=14 因子模式

Week 3-4 (P1):   MoE + RL 管线 ✅ 已完成
  ├── model/moe.py — MoE 多专家层 (222 lines)
  ├── model/rl_trainer.py — GRPO+PPO RL 训练器 (191 lines)
  └── model/kronos_model.py — MoE 可选后端

Week 5-8 (P2):   多资产 + 预训练 ⏳ 待执行

Week 5-8 (P2):   多资产 + 预训练
  ├── Week 5-6: 数据采集 (港股/美股)
  ├── Week 7: 预训练 base 模型
  └── Week 8: 全面评测 + 报告
```

### 优先级矩阵

| 方案 | 投入 | 预期 RankIC 提升 | 风险 | 优先级 |
|------|------|-----------------|------|--------|
| Qlib 集成 | 4h | 0 (基础建设) | 低 | 🔴 P0 |
| VPIN 因子注入 Transformer | 2h | +0.05-0.10 | 低 | 🔴 P0 |
| small (28M) 模型训练 | 8h | +0.03-0.05 | 低 | 🔴 P0 |
| MoE 多专家 | 10h | +0.02-0.05 | 中 | 🟠 P1 |
| GRPO→PPO RL 管线 | 12h | +0.05-0.10 (Sharpe) | 中 | 🟠 P1 |
| 多资产扩展 | 8h | +0.02-0.03 | 低 | 🟡 P2 |
| 预训练 Foundation Model | 30h | +0.05-0.10 | 高 | 🟡 P2 |

---

## 四、预期最终效果

### 4.1 目标指标

| 指标 | 当前 | Qlib+P0 后 | 全管线后 |
|------|------|-----------|---------|
| RankIC | +0.187 | **+0.25** | **+0.35** |
| Sharpe (年化) | ~1.2 | **~1.6** | **~2.2** |
| MaxDD | ~-15% | **~-10%** | **~-8%** |
| IR (Information Ratio) | 未评测 | **~1.0** | **~1.5** |
| 换手率 | — | **~30%/月** | **~25%/月** |

### 4.2 与 OSS 对比

| 维度 | 优化前 | 优化后 | MASTER | FinCast | neotema/Kronos |
|------|--------|--------|--------|---------|----------------|
| 回测框架 | 自建 | **Qlib** ✅ | Qlib ✅ | 自建 | 自建 |
| 模型规模 | 4M | **28M+** | ~5M | ~1B | 4M-500M |
| 因子融合 | 无 | **VPIN/DPIN** ✅ | 市场引导 | MoE 路由 | 无 |
| RL 优化 | LSTM only | **Transformer RL** ✅ | 无 | 无 | 无 |
| 多资产 | 2 类 | **A+H+美股** | 仅 A | 全球 | 45+ 交易所 |
| 预训练 | 无 | **Foundation** ✅ | 无 | 20B token ✅ | Foundation ✅ |
| 开源发布 | 无 | **HF Hub** | GitHub | HF Hub | HF Hub |

---

## 五、快速开始 (P0 立即执行)

```bash
cd /home/LLMs-from-scratch/projects/kronos-stock-predictor

# Step 1: Qlib 安装
pip install pyqlib
python -m qlib.run.get_data --region cn --interval 1d

# Step 2: 创建 Qlib 包装器
# 新建 model/qlib_kronos.py (参考上面代码)

# Step 3: 运行首轮 Qlib 回测
python -c "
from model.qlib_kronos import KronosQlibModel
import qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')
# ... 完整回测配置
"

# Step 4: 训练 small 模型
python train/train_predictor.py \
    --tokenizer_path outputs/tokenizer/best_model.pt \
    --data_dir data/semiconductor_v2/processed \
    --model_size small \
    --batch_size 4 \
    --epochs 10 \
    --output_dir outputs/predictor_small

# Step 5: 评测对比
python scripts/eval_accumulated.py \
    --predictor_path outputs/predictor_small/best_model.pt \
    --base_predictor outputs/predictor/best_model.pt
```
