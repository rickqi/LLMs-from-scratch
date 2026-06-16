# Kronos Stock Predictor — 项目评估

> 技术架构分析、实施策略与风险评估

---

## 一、方案选型分析

### 1.1 为什么选择 Kronos 架构

| 维度 | 传统时序模型 (LSTM/GRU) | 通用 TSFM (TimesFM/Moment) | Kronos |
|------|------------------------|---------------------------|--------|
| 金融适配 | 需大量特征工程 | 未针对金融优化 | 专为 K 线设计 |
| 离散化 | 直接回归连续值 | 大多连续建模 | BSQ 量化 → 离散 token |
| 多步预测 | 误差累积严重 | 较好 | 自回归 token 生成 |
| 概率预测 | 需额外建模 | 少数支持 | 内置采样 (top_p/temperature) |
| 可解释性 | 低 | 低 | 中（token 可分析） |

**结论**: Kronos 的 "离散化 + 自回归" 范式在金融场景中有天然优势。

### 1.2 模型规模选择

| 规模 | 适用场景 | GPU 需求 | 训练时间(估算) |
|------|---------|---------|---------------|
| **mini (4.1M)** | 原型验证、CPU 训练 | 无 | ~2h (CPU) |
| **small (28M)** | 单品种微调、研究 | 4GB+ | ~4h (T4) |
| **base (103M)** | 多品种预训练 | 8GB+ | ~24h (A100) |

**推荐策略**: mini → 验证架构正确性 → small → 单品种实验 → base → 多品种预训练。

---

## 二、架构决策记录

### 2.1 BSQ 量化 vs VQ-VAE

| 方案 | 优势 | 劣势 | 决策 |
|------|------|------|------|
| BSQ | 无码本崩溃、可扩展、球面几何适合金融尾部分布 | 实现较复杂 | ✅ **采用** |
| VQ-VAE | 实现简单、社区成熟 | 码本崩溃风险、大词表不可行 | ❌ |

### 2.2 层次化 token vs 单层 token

层次化 token (s1 粗 + s2 细) 将 2^20 的词表问题分解为 2×(2^10) 的预测问题，大幅降低 Transformer 输出头参数（从 1M → 2K）。

### 2.3 训练策略: 两阶段 vs 端到端

| 策略 | 稳定性 | 效果 | 决策 |
|------|--------|------|------|
| 两阶段 (Tokenizer → Predictor) | 高（解耦训练） | 好 | ✅ **采用** |
| 端到端 | 低（梯度冲突） | 理论最优 | ❌ |

### 2.4 数据源: Tushare vs akshare vs yfinance

| 源 | A 股覆盖 | 数据质量 | 费用 | 实时性 |
|----|---------|---------|------|--------|
| Tushare Pro | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 付费（积分） | T+1 |
| akshare | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 免费 | T+0 |
| yfinance | ⭐ | ⭐⭐⭐⭐ | 免费 | T+0 |

**推荐**: Tushare 为主 + akshare 备用。

### 2.5 分词器深度解析：从连续股价到离散 Token

#### 2.5.1 为什么需要分词？

传统时序模型（LSTM/Transformer 回归）直接预测连续价格值，存在三个根本问题：
1. **多步预测误差累积**：连续回归的输出误差逐步放大
2. **无法表达不确定性**：只输出点估计，无概率分布
3. **无法复用 LLM 工具箱**：top-k 采样、teacher forcing、KV cache 等都依赖离散输入

Kronos 的核心创新：**把连续 OHLCV 向量转化为离散 token**，从而将股价预测转化为类似 GPT 的 next-token prediction 问题。

#### 2.5.2 BSQ 量化机制详解

**Binary Spherical Quantizer（BSQ）** 是分词的核心引擎（`model/modules.py:161-273`）。与 VQ-VAE 使用 K-means 码本不同，BSQ 不维护任何码本向量——它用 **二值化（sign 函数）** 直接将连续向量压缩为二进制码：

```
输入:  z = [0.3, -1.2, 0.8, -0.5, ...]   (Encoder 输出, 20维)
           ↓ sign() 二值化
二进制: b = [+1,  -1,  +1,  -1,  ...]   (每个维度独立判定正负)
           ↓ 转换为整数索引
Token:  id = 0·2⁹ + 1·2⁸ + 0·2⁷ + ... = 逐位拼接为二进制数
```

**Straight-Through Estimator (STE)** 是让二值化可训练的关键：

| 阶段 | 行为 | 说明 |
|------|------|------|
| 前向传播 | `b = sign(z)` | 硬二值化，输出 ±1（不可导） |
| 反向传播 | `∂L/∂z = ∂L/∂b` | 梯度直接穿过，假装 sign() 是恒等函数 |

STE 使得离散的量化步骤在数学上"可微"，梯度可以从 Decoder 端一路传回 Encoder，端到端训练分词器。

**球面归一化**：量化后的向量做 L2 归一化，投影到超球面上。这一设计的金融意义在于——市场状态的"方向"比"幅度"更重要（牛市/熊市的区分不依赖于涨了多少，而是方向）。球面几何也天然适配金融数据的肥尾分布。

#### 2.5.3 为什么选 BSQ 而非 VQ-VAE？

| 对比维度 | VQ-VAE（传统） | BSQ（本项目） |
|---------|---------------|--------------|
| 码本存储 | 需存储 K 个码本向量（如 K=1024 × 768 维） | **无码本**——sign() 隐式定义码本 |
| 码本坍缩 | 常见——大量码字从不被激活，有效词表远小于名义词表 | **不存在**——每维独立二值化，所有 2^k 组合原则上可达 |
| 词表扩展 | 增加 K 需重新训练码本 | 增加 bit 数即可，天然可扩展 |
| 训练稳定性 | 码本更新需要 EMA，敏感 | 无需码本更新，更稳定 |
| 计算效率 | 最近邻搜索 O(K) | sign() O(1) |
| 金融适配 | K-means 假设球形簇，不适合肥尾分布 | 球面几何天然适配 |

#### 2.5.4 层次化 Token 设计：s1 粗 + s2 细

20 位二进制码被拆分为两个 10 位子码：

```
20位 BSQ 码:  [b₁ b₂ b₃ ... b₁₀ | b₁₁ b₁₂ ... b₂₀]
                   ↓                    ↓
                s1 (10 bit)          s2 (10 bit)
                词表: 2¹⁰ = 1024      词表: 2¹⁰ = 1024
```

**为什么要拆分？**

不拆分则词表 = 2²⁰ = 1,048,576，Transformer 输出层的 softmax 需要对 100 万类做分类——参数量巨大且训练困难。拆分后：
- 预测 s1：1024 类分类（粗粒度趋势）
- 条件预测 s2：1024 类分类（已知 s1 后的细节）

**两阶段重建目标保证语义分离**（`train/losses.py:9-28`）：

```python
# s1 必须能独立重建大致走势（迫使 s1 编码"粗"信息）
z_coarse = decoder(s1_only)
L_coarse = MSE(z_coarse, original_ohlcv)

# s1+s2 完整重建精确值（s2 编码"细"残差）
z_fine = decoder(s1 + s2)
L_fine = MSE(z_fine, original_ohlcv)

# BSQ 承诺损失（编码器输出靠近量化点）
L_bsq = MSE(z, quantize(z).detach()) + 0.25 * MSE(z.detach(), quantize(z))

Total = L_coarse + L_fine + λ * L_bsq
```

这个设计确保：即使只用 s1 重建，也能捕捉价格的主要趋势方向；s2 则补充精确幅度等细节信息。

#### 2.5.5 Encode/Decode 完整管线

**编码（OHLCV → Token）** `kronos_tokenizer.py:232-266`：

```
OHLCV (B, T, 6)
  → Linear(6 → d_model)                    # 特征嵌入
  → Encoder: n_enc_layers 个 TransformerBlock  # 时序上下文编码
  → Linear(d_model → 20)                   # 投影到码空间
  → BSQ: sign() + 球面归一化               # 离散化
  → 拆分 → s1_ids (B,T), s2_ids (B,T)     # 整数 ∈ [0, 1023]
```

**解码（Token → OHLCV）** `kronos_tokenizer.py:271-297`：

```
s1_ids, s2_ids (B, T)
  → indices_to_bits: 整数 → 二进制 → 双极 {-1,+1}
  → 拼接为 20 维向量
  → Linear(20 → d_model)
  → Decoder: n_dec_layers 个 TransformerBlock
  → Linear(d_model → 6)                    # 重建 OHLCV
```

#### 2.5.6 与文本 BPE 分词的对比

| 维度 | BPE（文本） | BSQ（Kronos） |
|------|------------|--------------|
| 分词方式 | 规则驱动，基于频率合并字节对 | **学习式**，Transformer 自编码器自动发现最优编码 |
| 码本 | 预训练固定（如 GPT-2 的 50,257 token） | 无显式码本，sign() 隐式定义 |
| 输入模态 | 离散字符 | 连续浮点向量（OHLCV） |
| 可逆性 | 不可逆（分词后无法完美重建原始文本） | 近似可逆（Tokenizer 训练目标就是重建） |
| 层次性 | 无（每个 token 独立） | s1/s2 两级层次（粗→细） |
| 训练需求 | 无需训练（用预训练词表） | 需要先训练 Tokenizer 自编码器 |

### 2.6 两阶段训练策略详解：解耦特征学习与序列建模

#### 2.6.1 "解耦"的具体含义

"解耦"指的是将模型学习拆分为两个独立优化目标：

| 阶段 | 学习目标 | 优化什么 | 类比 LLM |
|------|---------|---------|---------|
| **阶段 1：训练 Tokenizer** | 如何把连续 OHLCV 编码为好的离散 token | Tokenizer 自编码器的全部参数 | 相当于训练 BPE 词表（但 BPE 是规则的，这里是学习的） |
| **阶段 2：训练 Predictor** | 如何根据历史 token 序列预测未来 token | Predictor 的全部参数（Tokenizer **冻结**） | 相当于 GPT 预训练 |

两个阶段优化的是完全不同的能力：
- **阶段 1 解决"表征问题"**：什么样的离散编码能最好地表示一根 K 线的全部信息？（特征压缩与重建）
- **阶段 2 解决"时序问题"**：给定历史 K 线的 token 序列，下一根 K 线的 token 是什么？（模式识别与预测）

#### 2.6.2 阶段 1：训练 Tokenizer 自编码器

**目标**：学会将 6 维连续 OHLCV 向量压缩为 20 位二进制码，并能从二进制码重建回 OHLCV。

```
训练数据:  OHLCV 滑动窗口 (B, T, 6)
     ↓
Tokenizer.encode → s1_ids, s2_ids (离散 token)
     ↓
Tokenizer.decode → 重建 OHLCV'
     ↓
Loss = MSE(粗重建, 原始) + MSE(精重建, 原始) + λ·BSQ损失
```

**关键点**：
- 这一阶段**不涉及任何预测任务**——不预测未来，只学习"压缩-重建"
- Tokenizer 看到的是每个时间步的原始数据，目标是**信息无损压缩**
- 训练完成后，Tokenizer 成为 OHLCV ↔ Token 的固定转换器

**实测效果**（真实半导体数据，mini 模型）：
- 仅 3 epochs 即收敛到 val_loss = 0.136
- 说明 OHLCV → Token 的离散化高度有效——K 线的低维结构使得 20 bit 足以保留绝大部分信息

#### 2.6.3 阶段 2：冻结 Tokenizer，训练 Predictor

**目标**：在 Tokenizer 生成的离散 token 序列上做 next-token prediction。

```
训练数据:  OHLCV 滑动窗口 (B, T, 6)
     ↓
[冻结的 Tokenizer].encode → s1_ids, s2_ids    ← 不更新梯度
     ↓
shift by 1:  input = token[:-1], target = token[1:]  ← teacher forcing
     ↓
Kronos Predictor (GPT-style Transformer)
     ↓
Loss = CrossEntropy(s1_logits, s1_target) + CrossEntropy(s2_logits, s2_target)
```

**为什么冻结 Tokenizer？**

1. **梯度隔离**：如果不冻结，预测器的 cross-entropy 梯度会反向传播到 Tokenizer 的量化层。但量化（sign 函数）本身是不可导的——虽然 STE 能近似传梯度，但预测误差和重建误差的梯度方向经常冲突，导致训练不稳定。

2. **目标分离**：Tokenizer 的优化目标是"重建精度"（MSE），Predictor 的优化目标是"预测准确率"（CrossEntropy）。端到端训练会让两个目标互相干扰——为了好预测可能牺牲重建质量，反之亦然。

3. **效率**：冻结后，阶段 2 可以预先将所有训练数据一次性编码为 token，后续训练直接操作整数索引，跳过整个 Tokenizer 前向传播。

4. **可复现性**：同一个 Tokenizer 可以服务于不同规模（mini/small/base）的 Predictor，互不干扰。

#### 2.6.4 为什么不端到端训练？

| 问题 | 端到端 | 两阶段 |
|------|--------|--------|
| 梯度冲突 | 重建(MSE) + 预测(CE) 梯度方向矛盾 | ✅ 各自独立优化 |
| 训练稳定性 | sign() STE + 预测梯度 → 震荡 | ✅ 冻结后 Predictor 是标准 GPT 训练 |
| 超参敏感性 | 学习率/loss权重需精细调参 | ✅ 两阶段各自调参，互不影响 |
| 理论上限 | 理论最优（联合优化空间更大） | ⚠️ 可能次优（Tokenizer 不为预测任务优化） |
| 工程复杂度 | 高（需平衡多个 loss 项） | ✅ 低（分步调试） |

端到端在理论上可能达到更高上限（因为 Tokenizer 可以专门为预测任务优化编码方式），但实践中训练不稳定的问题使其难以收敛。两阶段策略以可复现性和稳定性换取理论上限的一小步损失——在当前 0.5M 参数的 mini 模型上已经足够。

#### 2.6.5 两阶段训练的类比

这个设计模式在深度学习中并不孤立：

| 领域 | 阶段 1（表征） | 阶段 2（任务） |
|------|---------------|---------------|
| **LLM** | BPE 分词器训练（或预训练词嵌入） | GPT 预训练 + 微调 |
| **语音** | wav2vec 自监督特征学习 | 下游 ASR 分类头 |
| **图像** | VQ-VAE/VQGAN 图像分词 | 隐藏 AR 模型图像生成 |
| **Kronos** | BSQ 自编码器 K 线分词 | GPT 式 next-token 预测 |

核心思想一致：**先学会"看"（表征学习），再学会"预测"（任务学习）**。

### 2.7 Tokenizer 训练数据规模分析

#### 2.7.1 单个训练样本的结构

每个训练样本是一个滑动窗口（`data/dataset.py:44`）：

```
窗口长度 = lookback_window(90) + predict_window(10) + 1 = 101 个时间步
特征维度 = 6 (open, high, low, close, vol, amt)
→ 单样本 shape: (101, 6) float32 = 101 × 6 × 4 bytes ≈ 2.4 KB
```

#### 2.7.2 训练数据规模（实测半导体数据集）

68 只半导体股票，2018-2026（~2000 交易日/只）：

```
每只股票可切窗口: 2000 - 101 + 1 ≈ 1,900 个
总可用窗口: 68 × 1,900 ≈ 129,200 个
```

Dataset 中有容量上限（`dataset.py:74-78`）：

```python
n_train_iter = 2000 × batch_size  # = 2000 × 50 = 100,000
n_val_iter   = 400 × batch_size   # = 400 × 50 = 20,000
n_samples    = min(target, len(indices))  # 取较小值
```

| 数据集 | 容量上限 | 实际可用窗口 | 实际使用 |
|--------|---------|------------|---------|
| 训练集 | 100,000 | ~129,200 | **100,000**（受上限约束） |
| 验证集 | 20,000 | ~16,000 | **~16,000**（受可用量约束） |

每个 batch 的显存占用：

```
batch_size=50 → (50, 101, 6) float32 ≈ 1.2 MB / batch
整个训练集 ≈ 100,000 × 2.4 KB ≈ 240 MB（纯数值）
```

#### 2.7.3 Tokenizer 产出物

**阶段 1 直接产出：模型权重（checkpoint）**

| 模型规模 | Tokenizer 参数量 | checkpoint 大小 |
|---------|-----------------|----------------|
| mini | ~0.5M | ~2 MB |
| small | ~2M | ~8 MB |
| base | ~8M | ~32 MB |

**阶段 1 的核心产出：编码能力（信息压缩）**

训练完成后，Tokenizer 的 `encode()` 方法将连续 OHLCV 压缩为离散 token：

```
输入:  (B, 101, 6) float32        每步 6 × 32 = 192 bits
                ↓ BSQ 编码
输出:  s1_ids (B, 101) int        每步 10 bits  ← 粗粒度趋势
       s2_ids (B, 101) int        每步 10 bits  ← 细粒度细节
                                每步合计 20 bits
```

| 维度 | 原始 (float32) | Token (s1+s2) | 压缩比 |
|------|---------------|--------------|--------|
| 每个时间步 | 192 bits | 20 bits | **9.6×** |
| 单个窗口 (101 步) | 19,392 bits ≈ 2.4 KB | 2,020 bits ≈ 253 B | **9.6×** |
| 10 万个窗口 | ~240 MB | **~25 MB** | **9.6×** |

#### 2.7.4 一根 K 线 → 一个 Token 对

```
时间步 t 的 OHLCV:               Token:
┌──────────────────────┐
│ open  = 12.50  ──┐   │
│ high  = 12.80  ──┤   │     s1_id = 731  (粗: 大致趋势方向)
│ low   = 12.30  ──┼─ 6维  →  s2_id = 204  (细: 精确形态细节)
│ close = 12.70  ──┤   │     = 一对整数 ∈ [0, 1023]
│ vol   = 1.2M   ──┤   │
│ amt   = 15.4M  ──┘   │
└──────────────────────┘
  192 bits (连续)         20 bits (离散)
```

#### 2.7.5 最小可行数据量

实测中真实数据仅 **3 epochs** 即收敛到 val=0.136（`ANALYSIS.md:150`），表明 Tokenizer 是轻量级自编码器，数据需求远低于 Predictor：

| 数据规模 | 股票数 | 天数/只 | 窗口数 | 预期效果 |
|---------|--------|--------|--------|---------|
| 最小可行 | 5-10 | 500+ | ~5,000 | 能收敛，重建质量一般 |
| **当前实测** | **68** | **~2,000** | **~100,000** | **3 epochs 收敛，val=0.136** |
| 推荐扩展 | 300+ | 2,000+ | 300,000+ | 更好的 token 表征，泛化更强 |

Tokenizer 只学"一根 K 线如何压缩为 20 bit"，不涉及时序预测——因此 68 只股票已远超需求。真正的数据瓶颈在阶段 2 的 Predictor（需更多时序多样性来学习跨品种、跨周期的预测模式）。

#### 2.7.6 完整数据流总结

```
原始 CSV (68 股票 × 2000 天 × 6 列)
  │  ≈ 68 × 2000 × 6 × 4 bytes ≈ 3.3 MB (纯数值)
  │
  ▼  preprocess_series() + build_dataset()
pickle 文件 (含归一化、均值、标准差、时间特征)
  │  ≈ ~15 MB (train) + ~3 MB (val) + ~3 MB (test)
  │
  ▼  StockDataset 滑动窗口采样
  │  每次抽取 (101, 6) ≈ 2.4 KB → batch (50, 101, 6) ≈ 1.2 MB
  │
  ▼  Tokenizer.encode()  ← 阶段 1 训练的成果
s1_ids + s2_ids: (50, 101) 整数对 ≈ 0.4 MB / batch
  │
  ▼  Predictor 训练 (阶段 2)
  │  输入: 整数 token 序列 → 标准 GPT 式训练
```

### 2.8 推理流程深度解析

#### 2.8.1 自回归推理完整流程

推理函数 `auto_regressive_inference`（`kronos_model.py:347-496`）执行以下流程：

```
输入: 历史 token 序列 (s1_ids, s2_ids) + 时间戳
  │
  ▼  循环 pred_len 次:
  │
  ├── ① decode_s1: Transformer 前向 → 取最后位置 s1_logits
  │   model.decode_s1(s1_buf, s2_buf, stamp)
  │   → s1_logits (S, T, 1024), context (S, T, d_model)
  │
  ├── ② 采样 s1: temperature → top-k → top-p → multinomial
  │   sample_from_logits(s1_logits[:, -1, :], T=0.6, top_p=0.9)
  │   → s1_sampled (S,)
  │
  ├── ③ decode_s2: 用 context + 采样的 s1 做条件预测
  │   model.decode_s2(context, s1_with_sampled)
  │   → s2_logits (S, T, 1024)
  │
  ├── ④ 采样 s2: 同样的 temperature/top-p 过滤
  │   sample_from_logits(s2_logits[:, -1, :])
  │   → s2_sampled (S,)
  │
  └── ⑤ 追加到 buffer，超出 max_context 则滑动窗口
      s1_buf = cat(s1_buf, s1_sampled) → 超长则截取最后 max_context 个
  │
  ▼  循环结束
  │
  ▼  tokenizer.decode(s1_pred, s2_pred) → OHLCV
  │
  ▼  多轨迹平均 → (pred_len, 6) 预测结果
```

#### 2.8.2 采样策略：Temperature / Top-k / Top-p

三种采样控制手段在 `sample_from_logits`（`kronos_model.py:75-107`）中依次执行：

```
原始 logits (1024 维)
  │
  ▼  Temperature 缩放
  logits = logits / T
  │  T < 1 → 更确定（趋向 argmax）
  │  T > 1 → 更随机（趋向均匀）
  │  T = 0.6（默认）→ 适度确定性，保留多样性
  │
  ▼  Top-k 过滤（默认 k=0，禁用）
  只保留概率最高的 k 个 token，其余设为 -inf
  │
  ▼  Top-p（nucleus）过滤（默认 p=0.9）
  按概率降序排列，累计概率 ≥ 0.9 后截断
  自适应：确定性强时选少数 token，不确定性高时选更多
  │
  ▼  Softmax → Multinomial 采样
```

**金融场景的采样意义**：股票未来不确定，不应该只输出一个确定性预测。通过 temperature + top-p 采样生成多条独立轨迹再平均，相当于对市场未来的**概率分布做蒙特卡洛估计**。

#### 2.8.3 多轨迹采样（sample_count）

```python
# 输入复制 sample_count 份，各自独立采样
s1_buf = x["s1_ids"].repeat(sample_count, 1)   # (S, T_ctx)
# ... 每步独立采样 ...
# 最终平均
averaged = np.mean(np.stack(results, axis=0), axis=0)  # (pred_len, d_in)
```

默认 `sample_count=5`：生成 5 条独立的未来 K 线轨迹，取平均作为最终预测。这降低了单次采样的随机性，类似于 ensemble 效果。

#### 2.8.4 滑动窗口上下文管理

```python
# 每步追加新 token
s1_buf = torch.cat([s1_buf, s1_sampled.unsqueeze(1)], dim=1)

# 超出 max_context 则截取最近窗口
if s1_buf.size(1) > max_context:
    s1_buf = s1_buf[:, -max_context:]
    s2_buf = s2_buf[:, -max_context:]
```

mini 模型 `max_context=2048`，可处理约 8 年的日线数据。超出时自动丢弃最早的 token，保持注意力计算量恒定。

### 2.9 Predictor 架构解析

Kronos Predictor（`kronos_model.py:150`）是 GPT 式 decoder-only Transformer，但针对离散金融 token 做了四项关键适配。

#### 2.9.1 层次化嵌入（HierarchicalEmbedding）

**文件**: `modules.py:280-310`

```python
class HierarchicalEmbedding(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model):
        self.emb_s1 = nn.Embedding(2**s1_bits, d_model)  # 1024 × d_model
        self.emb_s2 = nn.Embedding(2**s2_bits, d_model)  # 1024 × d_model

    def forward(self, token_ids):
        s1_ids, s2_ids = token_ids
        return self.emb_s1(s1_ids) + self.emb_s2(s2_ids)  # 相加融合
```

每个时间步有一对 token (s1, s2)，分别嵌入后**相加**为单个 d_model 维向量。这种设计让 Transformer 主干无需感知"层次"结构——它看到的始终是标准的 (B, T, d_model) 序列。

与 LLM 对比：GPT 用单一 `nn.Embedding(50257, d_model)`，Kronos 用两个 1024-size 嵌入表相加。

#### 2.9.2 时间特征嵌入（TemporalEmbedding）

**文件**: `modules.py:317-353`

```python
class TemporalEmbedding(nn.Module):
    # 5 个独立的嵌入表
    self.minute_emb  = nn.Embedding(60, d_model)    # 分钟
    self.hour_emb    = nn.Embedding(24, d_model)    # 小时
    self.weekday_emb = nn.Embedding(7, d_model)     # 星期几
    self.day_emb     = nn.Embedding(32, d_model)    # 日期
    self.month_emb   = nn.Embedding(13, d_model)    # 月份

    def forward(self, stamp):
        return minute + hour + weekday + day + month  # 五个嵌入相加
```

**为什么不用绝对位置编码？** 股票数据有强周期性（周内效应、月末效应、季节性），直接嵌入日历特征比学习一个递增的位置索引更合理。Transformer 通过这些时间嵌入自然学到"周一 vs 周五"、"月初 vs 月末"等模式。

最终输入 = `HierarchicalEmbedding(s1, s2) + TemporalEmbedding(stamp)`。

#### 2.9.3 条件依赖层（DependencyAwareLayer）

**文件**: `modules.py:360-382`

这是 s2 条件预测的核心机制——在预测 s2 之前，将已采样的 s1 信息注入上下文：

```python
class DependencyAwareLayer(nn.Module):
    def __init__(self, d_model, expansion=4):
        self.norm = RMSNorm(d_model)
        self.fuse = nn.Sequential(
            Linear(d_model, d_model * 4),   # 扩展
            GELU(),                          # 非线性
            Linear(d_model * 4, d_model),   # 压缩回
        )

    def forward(self, x, sibling_embed):
        # x: Transformer 上下文 (B,T,d_model)
        # sibling_embed: 采样的 s1 的嵌入 (B,T,d_model)
        fused = self.norm(x + sibling_embed)  # 融合上下文 + s1 信息
        delta = self.fuse(fused)              # MLP 变换
        return x + delta                      # 残差连接
```

**数据流**：`Transformer 上下文 x` + `已采样的 s1 嵌入` → MLP 融合 → 输出给 s2 头。这使得 s2 的预测**知道 s1 选了什么**，实现"先粗后细"的条件分解。

#### 2.9.4 双头输出（DualHead）

**文件**: `modules.py:389-409`

```python
class DualHead(nn.Module):
    self.head     = nn.Linear(d_model, 1024)  # s1 预测头
    self.cond_head = nn.Linear(d_model, 1024) # s2 条件预测头

    def forward(self, x):        return self.head(x)       # s1 logits
    def cond_forward(self, x):   return self.cond_head(x)  # s2 logits
```

两个独立的线性头，共享 Transformer 主干但参数独立。训练时同时计算两个交叉熵损失；推理时分两步调用（`decode_s1` → `decode_s2`）。

#### 2.9.5 训练时的 Teacher Forcing

训练 forward（`kronos_model.py:225-281`）中，s2 条件预测使用 ground-truth s1 而非采样值：

```python
if use_teacher_forcing:
    sibling_embed = self.embedding.emb_s1(s1_targets)  # 用真实 s1
else:
    s1_sampled = sample_from_logits(s1_logits.detach()) # 用采样 s1
    sibling_embed = self.embedding.emb_s1(s1_sampled)
```

Teacher forcing 使 s2 头在训练时始终看到正确的 s1，避免采样错误导致的误差累积。推理时则必须用采样的 s1（没有 ground truth）。

#### 2.9.6 Predictor 前向完整流程

```
(s1_ids, s2_ids) + stamp
  │
  ▼ ① 层次化嵌入
  x = emb_s1(s1_ids) + emb_s2(s2_ids)          # (B, T, d_model)
  │
  ▼ ② 时间嵌入
  x = x + time_emb(stamp)                       # + (minute+hour+weekday+day+month)
  │
  ▼ ③ Token Dropout
  x = token_drop(x)                             # 正则化
  │
  ▼ ④ Transformer 主干（n_layers × TransformerBlock）
  for layer in transformer:
      x = layer(x)                              # Pre-LN Attention + Pre-LN FFN
  │
  ▼ ⑤ 最终 RMSNorm
  x = norm(x)
  │
  ▼ ⑥ s1 预测
  s1_logits = head(x)                           # (B, T, 1024)
  │
  ▼ ⑦ 条件依赖（注入 s1 信息）
  sibling = emb_s1(s1_sampled_or_target)
  x2 = dep_layer(x, sibling)                    # x + MLP(LN(x + sibling))
  │
  ▼ ⑧ s2 条件预测
  s2_logits = cond_head(x2)                     # (B, T, 1024)
```

### 2.10 数据预处理管线

#### 2.10.1 滚动 Z-score 归一化（防未来泄露）

**文件**: `data/preprocessor.py:27-74`

```python
for i in range(n_rows):
    start = max(0, i - lookback + 1)
    window = values[start : i + 1]     # 只用 [i-lookback+1, i] 的数据
    mean_i = window.mean(axis=0)
    std_i  = window.std(axis=0)
    normed[i] = (values[i] - mean_i) / std_i
```

**关键设计**：每个时间步 i 的归一化统计（均值、标准差）只使用 `[i-lookback+1, i]` 窗口内的数据——**绝不触碰 i 之后的数据**。这从数据预处理层面消除了未来信息泄露。

归一化后裁剪到 `[-5.0, 5.0]`（`clip=5.0`），防止极端值（如停牌后复牌的跳空）破坏训练稳定性。

#### 2.10.2 滑动窗口采样

**文件**: `data/dataset.py:88-125`

```python
# 窗口大小
self.window = lookback_window(90) + predict_window(10) + 1 = 101

# 随机采样
def __getitem__(self, idx):
    symbol, start_idx = self.py_rng.choice(self.indices)  # 随机选(股票, 起始点)
    x = df.iloc[start_idx : start_idx + 101]              # 切出 101 天窗口

    # 只用 lookback 窗口做 Z-score
    past = x[:90]
    mean = past.mean(axis=0)
    std  = past.std(axis=0)
    x = (x - mean) / (std + 1e-5)
    x = clip(x, -5.0, 5.0)
```

每个 `__getitem__` 调用随机选取一只股票的一个起始位置，切出 101 天的 OHLCV 窗口。归一化统计**只用前 90 天**（lookback），后 11 天用同样的统计做变换——模拟真实场景中"用历史统计归一化未来数据"。

#### 2.10.3 时间特征工程

**文件**: `data/preprocessor.py:244-266`, `kronos_model.py:114-129`

```python
# 从 trade_date 提取 5 维时间特征
time_features = {
    "minute":  dt.dt.minute.values,     # 0-59（日线数据为 0）
    "hour":    dt.dt.hour.values,       # 0-23（日线数据为 15/9 等）
    "weekday": dt.dt.dayofweek.values,  # 0=周一, 6=周日
    "day":     dt.dt.day.values,        # 1-31
    "month":   dt.dt.month.values,      # 1-12
}
```

这些特征后续通过 `TemporalEmbedding` 映射为 d_model 维向量，加到 token 嵌入上。模型借此学习周期性模式（如周一效应、月末调仓等）。

#### 2.10.4 数据划分与防泄露

```python
# 按时间严格划分（非随机划分）
train_time_range = ["2015-01-01", "2022-12-31"]
val_time_range   = ["2023-01-01", "2023-12-31"]
test_time_range  = ["2024-01-01", "2025-06-30"]
```

时间序列必须按时间顺序划分——训练集永远在验证集之前，验证集在测试集之前。代码中通过日期 mask 实现严格隔离。

### 2.11 损失函数解析

#### 2.11.1 Tokenizer 损失：层次化重建 + 量化正则

**文件**: `train/losses.py:9-28`, `model/modules.py:254-258`

```python
def tokenizer_loss(x, x_recon_coarse, x_recon_fine, bsq_loss, lambda_quant=1.0):
    l_coarse = MSE(x_recon_coarse, x)   # 只用 s1 重建 → 迫使 s1 捕捉主趋势
    l_fine   = MSE(x_recon_fine, x)     # 用 s1+s2 重建 → s2 补充细节
    return l_coarse + l_fine + lambda_quant * bsq_loss
```

**BSQ 量化损失**（`modules.py:254-258`）：

```python
z_norm     = F.normalize(z, dim=-1)                              # 原始向量归一化
zhat_norm  = F.normalize(quantized * sqrt(embed_dim), dim=-1)    # 量化向量归一化
commit_loss = MSE(z_norm, zhat_norm.detach())                   # 编码器向量化点靠拢
            + beta * MSE(z_norm.detach(), zhat_norm)             # 量化点向编码器靠拢 (β=0.25)
```

两个 MSE 项构成**双向承诺**：编码器输出和量化点互相靠拢。detach 防止一项的梯度干扰另一项的优化目标。

**三项损失的分工**：

| 损失项 | 优化目标 | 作用 |
|--------|---------|------|
| L_coarse | s1-only → 原始 OHLCV | s1 编码"粗"信息（涨跌方向、趋势幅度） |
| L_fine | s1+s2 → 原始 OHLCV | s2 编码"细"残差（精确形态、波动细节） |
| BSQ_loss | z ↔ quantize(z) | 编码器输出靠近量化点，保证离散化质量 |

#### 2.11.2 Predictor 损失：双交叉熵

**文件**: `train/losses.py:31-68`

```python
def predictor_loss(s1_logits, s2_logits, s1_targets, s2_targets, s1_weight=1.0, s2_weight=1.0):
    loss_s1 = CrossEntropy(s1_logits.reshape(-1, 1024), s1_targets.reshape(-1))
    loss_s2 = CrossEntropy(s2_logits.reshape(-1, 1024), s2_targets.reshape(-1))
    total = s1_weight * loss_s1 + s2_weight * loss_s2
    return total, loss_s1, loss_s2
```

标准的 next-token prediction 交叉熵，分别对 s1 和 s2 各做一个 1024 类分类。权重默认 1:1，但可通过 `s1_weight`/`s2_weight` 调整——例如更关注粗趋势时增大 s1_weight。

**与 GPT 训练的对比**：

| 维度 | GPT 预训练 | Kronos Predictor |
|------|-----------|-----------------|
| 损失函数 | 单个 CrossEntropy | **双** CrossEntropy (s1 + s2) |
| 词表大小 | 50,257 | 2 × 1,024 |
| Teacher forcing | next-token | next-s1 → conditional next-s2 |
| Target 构造 | shift by 1 | 同一序列 shift by 1，但 s2 条件于 s1 |

#### 2.11.3 损失函数的层次化设计哲学

整个损失体系遵循一个核心原则：**粗粒度先验，细粒度后验**。

```
Tokenizer 层:
  L_coarse (s1单独重建)  → s1 学会"大局观"
  L_fine   (s1+s2重建)  → s2 学会"补细节"
  BSQ_loss             → 保证离散化不丢信息

Predictor 层:
  CE(s1)  → 先预测"大致是什么趋势"
  CE(s2|s1) → 在已知趋势的前提下预测"具体细节"
```

这种从粗到细的分解将一个困难的 2²⁰ 类预测问题转化为两个 2¹⁰ 类预测问题，不仅降低了计算复杂度，也符合金融市场的分层结构——先判断方向（涨/跌/震荡），再估计幅度。

---

## 三、风险清单

| 风险 | 严重度 | 概率 | 缓解 |
|------|--------|------|------|
| BSQ 训练不稳定 | 🔴 高 | 中 | EMA 更新、梯度裁剪、warmup LR |
| Tokenizer 重建质量差 | 🟡 中 | 中 | 先在小数据集验证、调 λ 参数 |
| 金融信噪比极低 | 🟡 中 | 高 | RankIC 评估、多轨迹采样、不过度承诺 |
| GPU 内存不足 | 🟡 中 | 低 | 从 mini 开始、batch_size 调优 |
| 过拟合 | 🟡 中 | 中 | 时序交叉验证、early stopping |
| Tushare token 不可用 | 🟢 低 | 低 | akshare 自动降级、可使用合成数据 |

---

## 四、评估指标

| 任务 | 指标 | 说明 |
|------|------|------|
| 价格预测 | RankIC | 预测排序与实际排序的相关性（金融场景首选） |
| 价格预测 | MSE / MAE | 点预测精度 |
| 方向预测 | Accuracy | 涨跌方向正确率 |
| 回测 | Sharpe Ratio | 风险调整后收益 |
| 回测 | Max Drawdown | 最大回撤 |
| 回测 | Annual Return | 年化收益率 |

---

## 五、扩展路径

| 阶段 | 行动 | 预期提升 |
|------|------|---------|
| 数据增强 | 添加分钟级/5分钟级 K 线 | 更多训练样本 |
| 多品种 | CSI300 → CSI800 → 全A股 | 更好的泛化 |
| 特征增强 | 添加技术指标 (MACD, RSI, 布林带) | 丰富输入信息 |
| 模型加深 | small → base → large (499M) | 更强表达能力 |
| 推理优化 | KV Cache、FlashAttention | 推理加速 |
| 集成策略 | 多模型投票/加权 | 降方差 |
| RL 微调 | GRPO 优化 Sharpe | 直接优化交易目标 |
| 部署 | ONNX/TensorRT 导出 | 生产部署 |

---

## 六、与 LLMs-from-scratch 的关系

本项目是 LLMs-from-scratch 学习路线的**进阶实战项目**，推荐完成以下章节后进行：

```
ch02 (文本数据处理) → 理解 tokenization 概念
ch03 (注意力机制) → 理解 Transformer 核心
ch04 (GPT 模型实现) → 理解 decoder-only 架构
ch05 (预训练) → 理解训练循环
ch07 (指令微调) → 理解 fine-tune 范式
→ 本项目: 将语言模型范式迁移到金融时序领域
```

关键差异对比:

| 维度 | LLMs-from-scratch (GPT) | 本项目 (Kronos) |
|------|------------------------|-----------------|
| 输入模态 | 文本 (BPE tokens) | K 线 (BSQ tokens) |
| 词表大小 | 50,257 (GPT-2) | 2¹⁰ = 1,024 (×2) |
| 嵌入层 | 单一 Embedding | 层次化 Embedding (s1+s2) |
| 输出层 | 单一 Linear | 双头条件输出 |
| 位置编码 | 绝对位置嵌入 | 时间特征嵌入 (minute/hour/day...) |
| 预测方式 | next-token | next-s1 + conditional-s2 |
| 评估 | Perplexity | RankIC / Sharpe |

---

## 七、实测结果评估（2026-06-15）

### 7.1 训练结果对比

| 指标 | 方案预估 | 合成数据实测 | 真实数据实测 | 评估 |
|------|---------|------------|------------|------|
| Tokenizer val (mini) | <0.5 | 0.2347 | **0.1360** | ✅ 远超预期 |
| Predictor val (mini) | <10 | 7.1063 | **4.3368** | ✅ 远超预期 |
| 回测 Sharpe | >0.5 | 1.45 | **1.96** | ✅ 优秀 |
| 回测年化收益 | >5% | 9.92% | **95.24%** | ⚠️ 需审慎解读 |
| 回测胜率 | >50% | 40.0% | **57.1%** | ✅ |
| 回测最大回撤 | <20% | -1.77% | -18.89% | ⚠️ 真实数据波动大 |

### 7.2 关键发现

1. **真实数据显著优于合成数据**：Predictor val loss 从 7.11 降至 4.34（↓39%），说明真实市场存在可学习的时序结构，BSQ tokenizer 有效捕获了这些模式。

2. **Tokenizer 收敛极快**：真实数据上 3 epochs 即达 val=0.136，表明 OHLCV→token 的离散化在半导体板块数据上高度有效。

3. **回测收益需审慎**：95.24% 年化收益部分源于半导体板块在测试期（2025-2026）的强趋势行情，且当前回测使用简化动量信号（非完整 Kronos 自回归预测），不代表模型真实 alpha。

4. **mini 模型已实用**：0.5M 参数即能达到 val=4.34，证明了 Kronos 架构在极小模型下的有效性，符合方案中 "mini → 验证 → 扩展" 的策略。

### 7.3 与方案预估的偏差

| 预估项 | 方案值 | 实测值 | 偏差原因 |
|--------|--------|--------|---------|
| Tokenizer 训练时间 | ~2h (CPU) | 22min (GPU) | GPU 可用，大幅加速 |
| 回测数据量 | 300 天 | 248 天 | 半导体数据从 2025-05 开始 |
| 品种数 | 300 (CSI300) | 69 (半导体) | 使用了半导体专属数据而非全市场 |

---

## 八、下一步建议

### 优先级排序

| 优先级 | 行动 | 预期效果 | 工作量 |
|--------|------|---------|--------|
| **P0** | 回测接入真实 Kronos 自回归预测 | 回测指标真实反映模型能力 | 2-3h |
| **P0** | 添加 RankIC 评估指标 | 更准确衡量预测排序能力 | 1h |
| **P1** | 扩大品种池：半导体 69 → CSI300/CSI800 | 更广泛的选股 alpha | 4h |
| **P1** | small 模型完整训练（5+ epochs） | 验证规模扩展收益 | 1h (GPU) |
| **P2** | 增加分钟级/5分钟级 K 线 | 更丰富的时序模式 | 8h |
| **P2** | 添加技术指标特征 (MACD, RSI, 布林带) | 增强输入信息量 | 3h |
| **P3** | base 模型预训练（103M, A100） | 验证大规模效果 | 24h+ |
| **P3** | Walk-forward 严格回测 + 行业中性化 | 接近实盘评估 | 8h |
| **P3** | 多模型集成 + 风险模型 | 降低过拟合 | 16h |

### 风险提示

- 当前回测信号为简化动量，非完整 Kronos 预测 → 真实 alpha 待验证
- 半导体板块高度相关，跨板块泛化能力未知
- mini 模型容量有限，复杂模式可能需要 small/base
- 95% 年化收益不可持续，实盘需考虑滑点、冲击成本、容量约束
