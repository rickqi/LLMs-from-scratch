# 第 05 篇：GPT 模型实现（第 4 章）

> 📖 基于《Build a Large Language Model (From Scratch)》视频笔记 + 章节内容
> 🎬 [GPT 模型实现视频](https://www.youtube.com/watch?v=YSAkgEarBGE)

---

## 一、本篇目标

**一句话**：从零实现完整的 GPT-2 架构，将前面章节的各个组件组装成可运行的 LLM。

读完本篇，你将：
- 理解 GPT-2 的完整架构：嵌入层 → TransformerBlock × N → 输出层
- 实现 LayerNorm、GELU、FeedForward、TransformerBlock 等核心组件
- 掌握 Pre-LayerNorm 与 Post-LayerNorm 的区别
- 理解 residual connection 对深层网络训练的关键作用
- 组装完整的 GPTModel 并验证参数量
- 了解 weight tying（权重绑定）的设计权衡

> ⏱️ 预计用时：3-4 小时

---

## 二、核心知识点

### 2.1 GPT 架构概览

GPT-2 的架构可以概括为：**Token Embedding + Positional Embedding + N 个 TransformerBlock + Final LayerNorm + Output Head**。

```
GPTModel:
├── tok_emb (nn.Embedding)    词元嵌入     vocab_size × emb_dim
├── pos_emb (nn.Embedding)    位置嵌入     context_length × emb_dim
├── drop_emb (nn.Dropout)     嵌入 Dropout
├── trf_blocks (nn.Sequential)  12 个 TransformerBlock
│   └── 每个 TransformerBlock:
│       ├── norm1 (LayerNorm)
│       ├── att (MultiHeadAttention)
│       ├── drop_shortcut (Dropout)
│       ├── norm2 (LayerNorm)
│       ├── ff (FeedForward)
│       └── drop_shortcut (Dropout)
├── final_norm (LayerNorm)
└── out_head (nn.Linear)      输出投影     emb_dim × vocab_size
```

**数据流**：
```
输入 token IDs (batch, seq_len)
  → tok_emb + pos_emb           # (batch, seq_len, 768)
  → drop_emb
  → 12 × TransformerBlock       # (batch, seq_len, 768)
  → final_norm
  → out_head                    # (batch, seq_len, 50257) = logits
```

> **位置编码演进**：GPT-2 使用的是**可学习的绝对位置嵌入**（`nn.Embedding(context_length, emb_dim)`）——每个位置学一个固定向量，直接加到 token 嵌入上。现代 LLM（LLaMA、Mistral、Gemma、Qwen）大多改用 **RoPE（旋转位置编码）**——不再往 token 向量里加位置信息，而是按 token 的位置将 Query 和 Key 向量旋转一个角度，天然编码相对位置，对长上下文泛化更好，且不增加参数。本书 `ch05/07_gpt_to_llama/` 补充目录展示了从 GPT 架构迁移到 LLaMA（含 RoPE）的完整实现。

**GPT-2-small 配置字典**：

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇量（GPT-2 BPE tokenizer）
    "context_length": 1024,  # 最大上下文长度
    "emb_dim": 768,          # 嵌入维度
    "n_heads": 12,           # 注意力头数
    "n_layers": 12,          # TransformerBlock 层数
    "drop_rate": 0.1,        # Dropout 比率
    "qkv_bias": False        # Q/K/V 是否使用 bias
}
```

> 这就是原始 GPT-2（124M 参数）的完整配置。本章的目标就是用 PyTorch 实现这个架构。

---

### 2.2 Layer Normalization

LayerNorm 的作用是**对每一层的输出做归一化**，使其均值为 0、方差为 1，让梯度流更稳定。

**为什么需要归一化？** 深层网络中，每层的输出分布可能剧烈变化（internal covariate shift），导致训练不稳定。LayerNorm 让每一层的输出分布保持一致，优化器可以更高效地工作。

**实现代码**：

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习缩放
        self.shift = nn.Parameter(torch.zeros(emb_dim))   # 可学习偏移

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

**关键点**：
- `eps = 1e-5`：防止除零的小常数
- `unbiased=False`：使用有偏方差估计（除以 N 而非 N-1），与 PyTorch 的 `nn.LayerNorm` 行为一致
- `scale` 和 `shift` 是可学习参数（即 γ 和 β），让模型有能力恢复到归一化前的表示

**Pre-LN vs Post-LN（演进视角）**：

2017 年原始 Transformer（Vaswani et al.）在每个子块**之后**做归一化（post-norm），浅模型没问题，深了（>10 层）就难训。现代 Transformer（GPT-2 起，LLaMA、Mistral、Gemma）普遍改在子块**之前**做归一化（pre-norm），这是让超深 Transformer 变得好训的关键改动之一。

| 类型 | 计算流程 | 特点 | 代表模型 |
|------|----------|------|---------|
| Post-LN | x → att → add → norm → ff → add → norm | 需要 warmup，训练不稳定 | 2017 Transformer |
| **Pre-LN** | x → **norm** → att → add → **norm** → ff → add | 训练更稳定，无需 warmup | GPT-2、LLaMA、Mistral |

```
Post-LN:  x → Attention(x) → x + Attention(x) → LayerNorm(...)
Pre-LN:   x → LayerNorm(x) → Attention(LN(x)) → x + Attention(LN(x))
```

> GPT-2 采用 Pre-LN，在注意力计算**之前**做归一化。这是现代 LLM 的主流选择。

**RMSNorm：归一化函数的进一步简化**：

很多现代开放模型（LLaMA、Mistral、Gemma、Phi）用更简单的 **RMSNorm** 替代标准 LayerNorm。原始 LayerNorm 做两件事——先把向量移向零点（减均值），再缩放大小（除标准差）。RMSNorm 砍掉移动只留缩放，经验上缩放承担了大部分收益，计算还更便宜：

```python
# RMSNorm 伪代码
def rms_norm(x):
    rms = sqrt(mean(x²))  # 只算均方根，不减去均值
    return x / rms * scale  # 不学习 shift 参数
```

| 类型 | 操作 | 参数 | 使用模型 |
|------|------|------|---------|
| LayerNorm | 减均值 + 除标准差 | γ（scale）+ β（shift） | GPT-2、BERT |
| **RMSNorm** | 仅除均方根 | γ（scale） | LLaMA、Mistral、Gemma、Phi |

> 💡 **为什么迁移到 RMSNorm？** 去掉 shift 和均值计算后，既减少了参数量又降低了计算开销，而层归一化的核心收益（控制数值范围）几乎不受影响。

---

### 2.3 GELU 激活函数

GELU（Gaussian Error Linear Unit）是 GPT 和 BERT 使用的激活函数，相比 ReLU 更平滑。

| 函数 | 公式 | 特点 |
|------|------|------|
| ReLU | `max(0, x)` | 简单高效，但 x < 0 时梯度为零（神经元"死亡"） |
| GELU | `x · Φ(x)` | 处处可导，更平滑，NLP 任务中效果更好 |

其中 Φ(x) 是标准正态分布的累积分布函数。

**实际使用中的近似公式**（计算更快）：

```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

**为什么 GPT 用 GELU 而不是 ReLU？**
- GELU 在 x ≈ 0 附近平滑过渡，而非 ReLU 的硬截断
- 对负值不直接归零，而是给一个小的权重，保留了更多信息
- 在 Transformer 架构的实验中，GELU 在 NLP 任务上通常优于 ReLU

**激活函数演进链：从 ReLU 到 SwiGLU**：

| 代数 | 函数 | 使用模型 | 特点 |
|------|------|---------|------|
| 第 1 代 | ReLU | 2017 Transformer | 简单，但 x<0 区域梯度为零 |
| 第 2 代 | GELU | GPT-2、BERT | 平滑过渡，负值保留少量信号 |
| **第 3 代** | **SwiGLU** | **LLaMA、Mistral、PaLM、Gemma** | 门控机制，FFN 中间层用两个权重矩阵 |

SwiGLU 是当下主流模型的默认选择。它的核心变化是在 FFN 的扩张步骤中引入一个**门控**（gating）机制：

```
传统 FFN:     output = W2 · GELU(W1 · x)
SwiGLU FFN:   output = W2 · (SwiGLU(W1 · x) ⊙ (W_gate · x))
                                   ↑非线性        ↑门控信号
```

门控让 FFN 有条件地激活某些信息通道，比单一路径的非线性更灵活。代价是多了一个权重矩阵，但这部分开销被更优的收敛速度补偿了。

> 💡 **为什么不直接用 GELU 的门控？** SwiGLU 用两个独立的权重矩阵（W1 和 W_gate）分别控制"非线性变换"和"是否让信息通过"，比 GELU 的单一矩阵 + 单一路径更精细。当前所有前沿开放权重模型（LLaMA 3、Mistral、Gemma 3、Qwen3）都使用 SwiGLU。

---

### 2.4 FeedForward 网络

FeedForward 是 TransformerBlock 中的第二个子层，由两个 Linear 层和中间的 GELU 组成。

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 768 → 3072
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 3072 → 768
        )

    def forward(self, x):
        return self.layers(x)
```

**为什么先扩大 4 倍再缩小回来？**

这是 Transformer 的经典设计（来自原始论文 "Attention Is All You Need"）：
1. **扩大**（768 → 3072）：在高维空间中学习更丰富的特征表示
2. **非线性变换**（GELU）：引入非线性能力
3. **缩小**（3072 → 768）：投影回原始维度，以便做 residual connection

这种"瓶颈-扩展-收缩"的设计让模型在每个位置上都能进行复杂的非线性变换。

---

### 2.5 TransformerBlock

TransformerBlock 是 GPT 架构的核心重复单元，组合了前面所有组件。

**内部结构（Pre-LN + Residual Connection）**：

```
输入 x
  │
  ├─→ LayerNorm(x) → MultiHeadAttention → Dropout ─→ +x (residual)
  │                                                      │
  │                                                      ▼
  └─→ LayerNorm(x') → FeedForward → Dropout ──────→ +x' (residual)
                                                         │
                                                         ▼
                                                       输出
```

**实现代码**：

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention 子层 + residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 残差连接

        # FeedForward 子层 + residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 残差连接

        return x
```

**Residual Connection（残差连接）为什么重要？**

1. **缓解梯度消失**：梯度可以通过 shortcut 直接回传到浅层，不需要经过所有变换
2. **让网络更容易学习**：模型可以学习恒等映射（跳过整个子层），也可以学习增量改进
3. **支持更深的网络**：没有 residual connection，12 层甚至 24 层的网络很难训练

> 比喻：如果注意力机制是引擎，residual connection 就是刹车——让模型在需要时可以"跳过"某个子层，不被迫使用每次变换的结果。

---

### 2.6 GPTModel 完整实现

将所有组件组装到一起：

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

**forward 流程逐步解析**：

| 步骤 | 操作 | 输入形状 | 输出形状 |
|------|------|----------|----------|
| 1 | `tok_emb(in_idx)` | (B, T) | (B, T, 768) |
| 2 | `pos_emb(arange(T))` | (T,) | (T, 768) → 广播为 (B, T, 768) |
| 3 | `tok_embeds + pos_embeds` | (B, T, 768) | (B, T, 768) |
| 4 | `drop_emb(x)` | (B, T, 768) | (B, T, 768) |
| 5 | `trf_blocks(x)` | (B, T, 768) | (B, T, 768) |
| 6 | `final_norm(x)` | (B, T, 768) | (B, T, 768) |
| 7 | `out_head(x)` | (B, T, 768) | (B, T, 50257) |

**关于 Weight Tying（权重绑定）**：

`out_head`（768 → 50257）和 `tok_emb`（50257 → 768）的权重矩阵形状互为转置。GPT-2 原始实现中**共享了这两个权重**，即 `out_head.weight = tok_emb.weight`。

- 本书实现中 `out_head` 使用 `bias=False`（无偏置），使得权重矩阵可以直接复用
- Weight tying 可以节省 50257 × 768 ≈ 38M 参数，总参数从 163M 降到 124M
- 这是一个设计选择，不是所有模型都用：Llama 3/3.1 不用，但 Llama 3.2 使用

---

### 2.7 GPT-2 各版本对比

OpenAI 发布了 4 个 GPT-2 版本，参数量从 124M 到 1.5B：

| 配置 | Small | Medium | Large | XL |
|------|-------|--------|-------|----|
| **参数量** | 124M | 355M | 774M | 1558M |
| n_layers | 12 | 24 | 36 | 48 |
| emb_dim | 768 | 1024 | 1280 | 1600 |
| n_heads | 12 | 16 | 20 | 25 |
| context_length | 1024 | 1024 | 1024 | 1024 |

> 本书使用 GPT-2 Small（124M），因为足够小，可以在普通 GPU 上训练。通过修改配置字典，同样的代码可以构建更大的版本。

---

## 三、动手练习：跑通第 4 章全流程

按以下步骤从零构建完整的 GPT-2 模型：

### 步骤 1：打开 notebook

```bash
cd LLMs-from-scratch
source .venv/bin/activate    # 激活虚拟环境
jupyter lab ch04/01_main-chapter-code/ch04.ipynb
```

### 步骤 2：实现 LayerNorm

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# 验证
ln = LayerNorm(emb_dim=5)
x = torch.randn(2, 3, 5)
out = ln(x)
print(f"输出均值: {out.mean(dim=-1)}")   # 预期: 接近 [0, 0, 0]
print(f"输出方差: {out.var(dim=-1)}")    # 预期: 接近 [1, 1, 1]
```

### 步骤 3：实现 GELU 激活函数

```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# 对比 GELU 和 ReLU
import matplotlib.pyplot as plt
gelu, relu = GELU(), nn.ReLU()
x = torch.linspace(-3, 3, 100)
# gelu(x) 在 x<0 时有平滑过渡，relu(x) 在 x=0 硬截断
```

### 步骤 4：构建 FeedForward 和 TransformerBlock

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

# 假设已有 MultiHeadAttention（第 3 章实现）
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
```

### 步骤 5：组装完整的 GPTModel

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

model = GPTModel(GPT_CONFIG_124M)

# 验证输出形状
batch = torch.randint(0, 50257, (2, 1024))  # 2 个样本，每个 1024 tokens
logits = model(batch)
print(f"logits shape: {logits.shape}")
# 预期: torch.Size([2, 1024, 50257])
```

### 步骤 6：检查参数量

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
# 预期: ~163,037,184 (163M)

# 如果使用 weight tying
tied_params = total_params - model.out_head.weight.numel()
print(f"Weight tying 后: {tied_params:,}")
# 预期: ~124,412,160 (124M)

# GPT-2 "124M" 之名来自 weight tying 后的参数量
```

✅ 全部通过 → 你已从零实现了完整的 GPT-2 架构！

---

## 四、常见问题

**Q: 为什么用 Pre-LN 而不是 Post-LN？**
A: Post-LN 中，梯度需要经过归一化层才能回传，深层网络中梯度容易消失。Pre-LN 将归一化放在子层之前，残差连接提供了一条"直通"的梯度路径。实验表明 Pre-LN 训练更稳定，不需要 learning rate warmup。GPT-2 及之后的大部分 LLM 都采用 Pre-LN。

**Q: GELU 和 ReLU 实际效果差多少？**
A: 在 Transformer 架构上，GELU 通常比 ReLU 好 1-2 个百分点的困惑度（perplexity）。GELU 在 x ≈ 0 附近平滑过渡，不直接截断负值，保留了更多梯度信息。虽然差距不算巨大，但 GELU 已成为 GPT、BERT 等 NLP 模型的事实标准。

**Q: Weight Tying 是什么？为什么要用？**
A: `tok_emb`（50257 → 768）和 `out_head`（768 → 50257）的权重矩阵形状互为转置。Weight tying 就是让它们共享同一组参数。好处：减少约 38M 参数（163M → 124M），节省显存和计算。坏处：限制了输出层和嵌入层各自的表达能力。是否使用取决于设计——Llama 3 不用，Llama 3.2 又用回来了。

**Q: Dropout rate 设多少合适？**
A: GPT-2 使用 0.1（10%）。这是经过大量实验验证的值。Dropout 太大（如 0.5）会导致欠拟合，太小（如 0.01）则几乎没有正则化效果。对于大模型，通常使用较小的 dropout（0.0-0.1），因为模型本身参数量大不容易过拟合。推理时 Dropout 自动关闭（`model.eval()`）。

**Q: 为什么 FeedForward 要先扩大 4 倍再缩小回来？**
A: 这是 Transformer 原始论文的设计。扩大到更高维度让模型能在更丰富的特征空间中学习非线性变换，然后再投影回原始维度。这种"扩展-压缩"结构增加了模型的表示能力。4x 是经验值，一些现代架构（如 Llama）会使用不同的比率。

**Q: `qkv_bias=False` 是什么意思？**
A: 在 MultiHeadAttention 中，Q、K、V 的线性映射默认不带 bias。原始 GPT-2 就是如此设计。不带 bias 可以略微减少参数量，且对性能影响很小。注意：GPT-2 的官方权重文件中 QKV bias 确实为 False，加载预训练权重时需要匹配。

---

## 五、考试卡片

> 学完第 4 章后，用这些卡片自测。遮住答案，看问题能不能答出来。

---

<details>
<summary><b>Q11: GPTModel 由哪些组件组成？画出结构。</b></summary>

**A:**

```
GPTModel:
├── tok_emb (nn.Embedding)     词元嵌入      vocab_size × emb_dim
├── pos_emb (nn.Embedding)     位置嵌入      context_length × emb_dim
├── drop_emb (nn.Dropout)
├── trf_blocks (nn.Sequential)  12 个 TransformerBlock
│   └── TransformerBlock:
│       ├── norm1 (LayerNorm)
│       ├── att (MultiHeadAttention)
│       ├── drop_shortcut (Dropout)
│       ├── norm2 (LayerNorm)
│       ├── ff (FeedForward)
│       └── drop_shortcut (Dropout)
├── final_norm (LayerNorm)
└── out_head (nn.Linear)       输出投影      emb_dim → vocab_size (bias=False)
```

每个 TransformerBlock 内部使用 **Pre-LayerNorm + 残差连接**。

</details>

---

<details>
<summary><b>Q12: Pre-LayerNorm 和 Post-LayerNorm 有什么区别？本书用哪种？</b></summary>

**A:**

```
Post-LN (原版 Transformer):  x → att → add → norm → ff → add → norm
Pre-LN  (GPT-2 / 本书):      x → norm → att → add → norm → ff → add
```

- **Post-LN**：归一化在残差连接之后，梯度需要经过 norm 层回传，训练不稳定，需要 warmup
- **Pre-LN**：归一化在子层之前，残差连接提供梯度"直通"路径，训练更稳定

**本书（GPT-2）使用 Pre-LN**，这也是现代 LLM 的主流选择。

</details>

---

<details>
<summary><b>Q13: GELU 和 ReLU 激活函数的区别？为什么 GPT 用 GELU？</b></summary>

**A:**

| 函数 | 公式 | 特点 |
|------|------|------|
| ReLU | `max(0, x)` | 简单高效，但 x < 0 时梯度为零（神经元"死亡"） |
| GELU | `x · Φ(x)` | 处处可导，更平滑，NLP 任务中效果更好 |

GELU（Gaussian Error Linear Unit）在 NLP 任务中表现优于 ReLU，是 GPT/BERT 标配。

- 在 x ≈ 0 处平滑过渡（ReLU 硬截断）
- 对负值不直接归零，保留更多信息
- 近似计算：`0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`

</details>

---

<details>
<summary><b>Q14: 为什么 GPTModel 的输出头 out_head 不使用 bias？</b></summary>

**A:**

`out_head` 的权重与词元嵌入 `tok_emb` 共享（权重绑定 / weight tying）。不带 bias 使得权重矩阵可以直接复用：
- `tok_emb` 的形状：50257 × 768（词汇表 → 嵌入空间）
- `out_head` 的形状：768 × 50257（嵌入空间 → 词汇表）

两者互为转置，共享后减少参数量（163M → 约 124M），且不损失性能。

> 注意：本书实现中并未真正执行权重共享（`out_head.weight = tok_emb.weight`），但使用了 `bias=False` 为可能的共享做准备。GPT-2 "124M" 的参数量名称来自包含 weight tying 的计算。

</details>

---

## 六、关键文件速查

| 想看什么 | 打开 |
|---------|------|
| 第 4 章完整 notebook | `ch04/01_main-chapter-code/ch04.ipynb` |
| GPT 模型精简版 | `ch04/01_main-chapter-code/gpt.py` |
| 练习答案 | `ch04/01_main-chapter-code/exercise-solutions.ipynb` |
| 前序章节代码 | `ch04/01_main-chapter-code/previous_chapters.py` |
| FLOPs 性能分析 | `ch04/02_performance-analysis/flops-analysis.ipynb` |
| KV Cache 实现 | `ch04/03_kv-cache/` |
| GQA 实现 | `ch04/04_gqa/` |
| MLA 实现 | `ch04/05_mla/` |
| SWA 实现 | `ch04/06_swa/` |
| MoE（Mixture-of-Experts） | `ch04/07_moe/` |
| Gated DeltaNet | `ch04/08_deltanet/` |
| DeepSeek Sparse Attention | `ch04/09_dsa/` |
| Cross-layer KV Sharing | `ch04/10_kv-sharing/` |
| GPT 视频字幕 | `youtube/07-GPT模型实现.md` |

---

## 七、补充资源

学完主线后，按需深入：

```
ch04/02_performance-analysis/   ← FLOPs 分析：理解模型计算瓶颈
ch04/03_kv-cache/               ← KV Cache：推理加速关键技术
ch04/04_gqa/                    ← Grouped-Query Attention（Llama 4/Qwen3 使用）
ch04/05_mla/                    ← Multi-Head Latent Attention（DeepSeek V3 使用）
ch04/06_swa/                    ← Sliding Window Attention（Gemma 3 使用）
ch04/07_moe/                    ← Mixture-of-Experts 稀疏专家模型
ch04/08_deltanet/               ← Gated DeltaNet 线性注意力变体
ch04/09_dsa/                    ← DeepSeek Sparse Attention
ch04/10_kv-sharing/             ← Cross-layer KV Sharing（Gemma 4 使用）
```

> **推荐优先级**：先完成主线 + 练习 → `03_kv-cache`（推理优化必备） → `04_gqa`（现代 LLM 标配） → `07_moe`（前沿架构） → 其余按兴趣。

---

## 上一篇

第 04 篇：注意力机制 → [04-注意力机制.md](04-注意力机制.md)

## 下一篇

第 06 篇：预训练 → [06-预训练.md](06-预训练.md)
