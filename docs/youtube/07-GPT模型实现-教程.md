# GPT 模型架构实现教程

> 🎬 基于 Sebastian Raschka《Build a Large Language Model (From Scratch)》配套视频
> 📺 [原视频](https://www.youtube.com/watch?v=YSAkgEarBGE) | ⏱️ 约 52 分钟

---

## 这个视频讲什么？

这是第 4 章的核心实现视频。Sebastian 带领我们从零搭建完整的 GPT-2 模型架构——先展示"骨架"（Dummy GPT），然后逐一填充每个组件：LayerNorm、GELU 激活函数、FeedForward 网络、Shortcut 残差连接、TransformerBlock，最终组装出完整的 GPTModel。如果把注意力机制比作汽车的发动机，这一章就是安装车轮、方向盘和座椅。

---

## 一、GPT 架构全景与配置

### 1.1 从 Dummy GPT 开始

在填充细节之前，先看整体架构。Dummy GPT 用占位符表示各个组件：

```python
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
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

### 1.2 GPT-2 124M 配置字典

```python
GPT_CONFIG_124M = {
    "vocab_size":    50257,  # GPT-2 tokenizer 的词汇表大小
    "context_length": 1024,  # 上下文窗口长度
    "emb_dim":         768,  # 嵌入维度
    "n_heads":           12, # 多头注意力的头数
    "n_layers":          12, # Transformer 块的层数
    "drop_rate":        0.1, # Dropout 比率
    "qkv_bias":       False, # Q/K/V 线性层是否使用 bias
}
```

### 1.3 数据流全景图

```
Token IDs (输入文本)
    │
    ▼
┌──────────────────────────┐
│  Token Embedding         │  vocab_size → emb_dim (768)
│  Positional Embedding    │  context_length → emb_dim (768)
│  x = tok_emb + pos_emb  │
│  Dropout                 │
└──────────┬───────────────┘
           │
    ┌──────▼──────┐
    │  Transformer│ ×12 层
    │  Block      │
    │  (见第四节)  │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  LayerNorm  │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Linear     │  emb_dim (768) → vocab_size (50257)
    │  (输出头)    │
    └──────┬──────┘
           │
           ▼
    Logits: [batch, seq_len, 50257]
```

### 1.4 输出维度解读

输入 batch 为 `[2, 4]`（2 个样本，每样本 4 个 token），模型输出 shape 为 `[2, 4, 50257]`：

| 维度 | 含义 | 值 |
|------|------|----|
| dim=0 | batch 大小 | 2 |
| dim=1 | 序列长度（token 数） | 4 |
| dim=2 | 词汇表大小（logits） | 50257 |

每个 token 被表示为 768 维向量（中间表示），输出层将其映射回 50257 维（每个词汇表项的得分）。这些得分称为 **logits**——未经 softmax 的原始分数。第 5 章将讲解如何从 logits 选择下一个 token。

---

## 二、Layer Normalization（层归一化）

### 2.1 为什么需要归一化？

在深度网络中，某一层的输出值可能非常大或非常小，导致：
- **梯度消失**：梯度信号逐层衰减，底层几乎无法更新
- **梯度爆炸**：梯度信号逐层放大，训练不稳定

LayerNorm 的目标：将每层的输出归一化为 **均值=0、方差=1**，使优化过程更稳定。

### 2.2 为什么不用 Batch Normalization？

| 特性 | Batch Norm | Layer Norm |
|------|-----------|------------|
| 归一化方向 | 跨样本（dim=0） | 跨特征（dim=-1） |
| 依赖 batch 大小 | 是（小 batch 不稳定） | 否 |
| 多 GPU 分布式训练 | 不友好 | 友好 |
| LLM 中的使用 | 不推荐 | 标准选择 |

LayerNorm 沿**特征维度**（最后一个维度）归一化，与 batch 大小无关，因此更适合 LLM 场景。

### 2.3 手动实现归一化步骤

```python
import torch

# 模拟一个 batch: 2 个样本，每个 5 维
torch.manual_seed(123)
batch = torch.randn(2, 5)

# 第一步：计算均值（沿最后一个维度）
mean = batch.mean(dim=-1, keepdim=True)

# 第二步：计算方差
var = batch.var(dim=-1, keepdim=True, unbiased=False)

# 第三步：归一化（减均值，除标准差）
normed = (batch - mean) / torch.sqrt(var)

# 验证：归一化后的均值约等于 0，方差约等于 1
print(normed.mean(dim=-1, keepdim=True))  # ≈ [0, 0]
print(normed.var(dim=-1, keepdim=True))    # ≈ [1, 1]
```

`keepdim=True` 保持原始维度，使 broadcasting 正常工作。`dim=-1` 比硬编码 `dim=1` 更通用。

### 2.4 LayerNorm 完整实现

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5                              # 防止除以零的小值
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可训练的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可训练的偏移参数

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

### 2.5 scale 和 shift 参数的意义

| 参数 | 初始值 | 作用 |
|------|--------|------|
| `scale` | 全 1 | 乘以 1 = 不改变。但网络可以学习不同值，甚至可以"撤销"归一化 |
| `shift` | 全 0 | 加上 0 = 不改变。网络可以学习加回均值，"撤销"中心化 |

**关键洞察**：归一化是强制的，但 scale/shift 给了网络"退出"的选择——如果网络发现归一化不利于学习，它可以学到 `scale = std`、`shift = mean`，完全撤销归一化。

`eps`（epsilon = 1e-5）是一个安全常数，加在方差上防止除以零。

> `unbiased=False` 是一个实现细节：PyTorch 的默认方差计算使用 Bessel 校正（除以 N-1），而原始 GPT-2 的 TensorFlow 实现使用总体方差（除以 N）。设 `unbiased=False` 以匹配原始行为。当维度为 768 时，两者差异可忽略。

---

## 三、GELU 激活函数

### 3.1 为什么需要非线性激活？

纯线性层的组合仍然是线性函数——无论堆叠多少层，网络只能学习线性映射。非线性激活函数让网络能够学习复杂的特征表示。

常见的非线性激活函数：Sigmoid、Tanh、ReLU、GELU、SwiGLU 等。

### 3.2 GELU vs ReLU

```
输出
  │        ReLU (阶跃)
  │       /
  │      /
  │     /    GELU (平滑曲线)
  │    /    /
  │   /    /
  │  /    .   <-- GELU 在零点附近更平滑
  │ .    /
  │.    /
  ──────────────── 输入
  │ 0
```

| 特性 | ReLU | GELU |
|------|------|------|
| 负值处理 | 直接截断为 0 | 平滑过渡（小负值保留） |
| 零点处梯度 | 不连续 | 连续可导 |
| 优化性质 | 一般 | 更好（平滑梯度） |
| GPT-2 使用 | 否 | 是 |

### 3.3 GELU 的两种形式

GELU 有精确公式和近似公式两种实现。原始 GPT-2 使用 **近似公式**（出于 TensorFlow 中的计算效率考虑）：

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

这就是论文《Gaussian Error Linear Units》中提出的近似函数。

> GELU 的具体选择不是最关键的——只要使用某种非线性激活函数即可。后来的 Llama 使用 SwiGLU，也可以正常训练。我们使用 GELU 是为了精确匹配原始 GPT-2 架构，以便第 5 章加载 OpenAI 的预训练权重。

---

## 四、FeedForward 前馈网络

### 4.1 FeedForward 模块

FeedForward 是 TransformerBlock 内部的一个小型神经网络，位于注意力层之后：

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

### 4.2 "扩展-收缩"结构

```
输入: [batch, seq, 768]
         │
         ▼
   Linear(768 → 3072)    ← 扩展 4 倍
         │
         ▼
       GELU()
         │
         ▼
   Linear(3072 → 768)    ← 缩回原尺寸
         │
         ▼
输出: [batch, seq, 768]
```

这种"扩展-收缩"（瓶颈的逆形式）的设计意图：
1. **增加参数量**：扩展层引入大量可训练参数
2. **特征提取**：在更高维空间中学习更丰富的特征表示
3. **信息压缩**：最终缩回原尺寸，保留最有用的信息

**为什么是 4 倍？** 这是原始 GPT-2 论文作者选择的超参数。它可以是 2 倍、8 倍或任何值，但必须与原始实现一致才能加载 OpenAI 的预训练权重。

---

## 五、Shortcut Connections（残差连接）

### 5.1 什么是残差连接？

残差连接（也叫 Skip Connection）来自 2015 年的 ResNet 论文。核心思想极其简单：

```
输入 x ──────────────────────┐
  │                           │
  ▼                           │
┌──────────┐                  │
│  Linear   │                  │
│  GELU     │                  │
│  Linear   │                  │
└────┬─────┘                  │
     │                        │
     ▼                        ▼
    F(x)  +  x  =  输出
```

**公式**：`output = F(x) + x`（将输入直接加到输出上）

### 5.2 为什么残差连接有效？

在深层网络的反向传播中，梯度信号会逐层衰减。残差连接提供了一条"旁路"：

| 场景 | 无残差连接 | 有残差连接 |
|------|-----------|-----------|
| 梯度信号 | 0.13 → 逐层更弱 | 1.32 → 2.63 → 2.2（强得多） |
| 子模块失效时 | 后续所有层都受影响 | 通过残差路径保留信息 |
| 网络可以学到 | 必须依赖每层 | 可以"跳过"无用的层 |

**直觉理解**：残差连接是一个"安全网"——即使某个子模块没学到有用的东西，梯度信号仍能通过旁路传递，网络整体不会崩溃。

### 5.3 梯度对比实验

```python
# 不使用残差连接的深度网络：梯度逐渐消失
# 梯度值: 0.0018 → 0.0016 → 0.0013 → ...

# 使用残差连接的深度网络：梯度保持稳定
# 梯度值: 1.32 → 2.63 → 2.20 → ...
```

---

## 六、TransformerBlock（完整组装）

### 6.1 架构图

```
输入 x ─────────────────────┐
  │                          │
  ▼                          │
┌──────────┐                 │
│ LayerNorm │                 │
└────┬─────┘                 │
     ▼                       │
┌──────────────────┐         │
│ MultiHeadAttention│         │
│ (Masked, 12 heads)│         │
└────┬─────────────┘         │
     ▼                       │
   Dropout                   │
     │                       │
     ▼                       │
  x + shortcut ──────────────┘  (残差连接 1)
     │
     ├── 保存为 shortcut ─────┐
     ▼                        │
┌──────────┐                  │
│ LayerNorm │                  │
└────┬─────┘                  │
     ▼                        │
┌──────────┐                  │
│ FeedForward│                 │
│ (Linear→  │                  │
│  GELU→    │                  │
│  Linear)  │                  │
└────┬─────┘                  │
     ▼                        │
   Dropout                    │
     │                        │
     ▼                        │
  x + shortcut ───────────────┘  (残差连接 2)
     │
     ▼
   输出
```

### 6.2 TransformerBlock 完整代码

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
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # ---- 注意力子块 + 残差连接 ----
        shortcut = x
        x = self.norm1(x)          # Pre-Norm（先归一化）
        x = self.att(x)            # 多头注意力
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut           # 残差连接

        # ---- 前馈子块 + 残差连接 ----
        shortcut = x
        x = self.norm2(x)          # Pre-Norm
        x = self.ff(x)             # FeedForward
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut           # 残差连接

        return x
```

### 6.3 Pre-Norm vs Post-Norm

本实现使用 **Pre-Norm**（先 LayerNorm，再进入子层）。这是 GPT-2 的原始设计，训练更稳定。

| 模式 | 顺序 | 代表模型 |
|------|------|---------|
| Pre-Norm | Norm → Sublayer → Add | GPT-2、GPT-3、大多数现代 LLM |
| Post-Norm | Sublayer → Add → Norm | 原始 Transformer |

---

## 七、完整 GPTModel

### 7.1 最终代码

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

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

### 7.2 数据流详解

```
输入: "Hello, I am" → tokenizer → [15496, 11, 314, 716]
                                              │
                                              ▼
                                  Token Embedding (50257×768)
                                              +
                                  Positional Embedding (1024×768)
                                              │
                                          Dropout(0.1)
                                              │
                                    ┌─────────▼─────────┐
                                    │ TransformerBlock 1 │
                                    │ TransformerBlock 2 │
                                    │       ...          │
                                    │ TransformerBlock 12│
                                    └─────────┬─────────┘
                                              │
                                        LayerNorm(768)
                                              │
                                     Linear(768→50257)
                                              │
                                              ▼
                              Logits: [1, 4, 50257]
                              (每个位置有 50257 个得分)
```

### 7.3 为什么输入层和输出层的维度是"对称"的？

- **输入层**（Token Embedding）：`vocab_size (50257) → emb_dim (768)`，将 token ID 映射到嵌入空间
- **输出层**（Linear）：`emb_dim (768) → vocab_size (50257)`，从嵌入空间映射回词汇表

这种"编码-解码"结构使得模型可以将中间表示转换回具体的 token 预测。

### 7.4 模型参数量

使用 GPT_CONFIG_124M 配置初始化模型后：

```python
model = GPTModel(GPT_CONFIG_124M)
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,.0f}")  # ≈ 163 million
```

> 实际参数量约 1.63 亿（比名称中的 124M 多），因为原始 GPT-2 论文计算参数量时去除了部分层。这不影响功能。

---

## 关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| LayerNorm | 沿特征维度归一化为均值0、方差1，含可学习的 scale 和 shift 参数 |
| GELU | 比 ReLU 更平滑的非线性激活函数，GPT-2 使用其近似公式 |
| FeedForward | Linear→GELU→Linear 的"扩展-收缩"结构，中间扩展 4 倍 |
| Shortcut Connection | 将输入直接加到输出上（残差连接），防止梯度消失 |
| TransformerBlock | 注意力层 + 前馈层，各自带 LayerNorm + Dropout + 残差连接 |
| Pre-Norm | 先 LayerNorm 再进入子层，比 Post-Norm 训练更稳定 |
| Logits | 输出层未经 softmax 的原始分数，shape 为 [batch, seq, vocab_size] |
| CFG | Configuration 的缩写，Python 字典，存储模型超参数 |
| Dropout | 训练时随机丢弃部分神经元输出，防止过拟合 |

---

## 常见问题

**Q1: 为什么必须精确匹配原始 GPT-2 的所有超参数？**

A: 第 5 章会从 OpenAI 下载预训练权重并加载到我们的模型中。权重的 shape 必须完全匹配——如果我们的 FeedForward 中间层是 3 倍而不是 4 倍，权重文件就无法加载。

**Q2: `unbiased=False` 在 LayerNorm 中重要吗？**

A: 对于 768 维的 embedding 来说，Bessel 校正（N vs N-1）的差异可以忽略不计。设置 `unbiased=False` 只是为了精确匹配原始 GPT-2 在 TensorFlow 中的实现行为。

**Q3: 为什么使用 GELU 而不是 ReLU？**

A: GELU 在零点附近更平滑，梯度更连续，优化性质更好。但具体选择哪个激活函数不是最关键的——Llama 使用 SwiGLU 也训练得很好。关键是使用某种非线性激活。

**Q4: Dropout 在哪里使用？什么时候开启/关闭？**

A: Dropout 在三个位置使用：embedding 层之后、注意力之后、前馈网络之后。训练时开启（`model.train()`），推理时关闭（`model.eval()`）。这是 PyTorch 的 nn.Dropout 自动处理的。

**Q5: Pre-Norm 和 Post-Norm 有什么区别？**

A: Pre-Norm（先归一化再计算）是 GPT-2 的选择，训练更稳定，无需学习率预热。Post-Norm（先计算再归一化）是原始 Transformer 的设计，理论上表现力更强但训练更困难。

---

## 下一步

模型架构已经搭建完毕，但它现在输出的是随机值。下一步是**预训练**——用大量文本数据训练模型，使其学会预测下一个 token。参见下一节教程：[08-预训练-教程.md](08-预训练-教程.md)。

---

## 关键文件速查

| 想看什么 | 打开 |
|---------|------|
| 完整 GPT 模型代码（独立脚本） | `ch04/01_main-chapter-code/gpt.py` |
| 带详细注释的 notebook | `ch04/01_main-chapter-code/ch04.ipynb` |
| 多头注意力实现（第 3 章代码） | `ch03/01_main-chapter-code/ch03.ipynb` |
| 数据加载器（第 2 章代码） | `ch02/01_main-chapter-code/dataloader.ipynb` |
| 性能分析（FLOPs） | `ch04/02_performance-analysis/flops-analysis.ipynb` |
| KV Cache 实现 | `ch04/03_kv-cache/` |
