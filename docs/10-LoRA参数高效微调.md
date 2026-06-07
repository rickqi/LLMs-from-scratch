# 第 10 篇：LoRA 参数高效微调（附录 E）

> 📖 基于《Build a Large Language Model (From Scratch)》附录 E
> 🔗 前置知识：第 6 章（分类微调）、第 7 章（指令微调）
> 💡 LoRA 是目前最流行的参数高效微调方法，几乎所有现代 LLM 微调都使用它

---

## 一、本篇目标

**一句话**：从零实现 LoRA（Low-Rank Adaptation），用不到 2% 的可训练参数完成与全参数微调相当的效果。

读完本篇，你将：
- 理解 LoRA 的数学原理（低秩分解）
- 实现 `LoRALayer`、`LinearWithLoRA`、`replace_linear_with_lora` 三个核心组件
- 对比全参数微调 vs LoRA 微调的参数量和性能
- 掌握 rank 和 alpha 超参数的含义与选择策略

> ⏱️ 预计用时：1-2 小时

---

## 二、核心知识点

### 2.1 LoRA 的动机与原理（Section E.1）

**问题**：全参数微调需要更新模型的所有参数。以 GPT-2 small 为例，124M 个参数全部参与训练，对显存和算力要求极高。

**LoRA 的核心洞察**：预训练权重矩阵的更新量 ΔW 具有低"内在秩"（intrinsic rank）——即使 ΔW 是一个高维矩阵，其有效信息可以用两个小矩阵的乘积来近似。

**数学公式**：

全参数微调的权重更新：

$$W_{\text{updated}} = W + \Delta W$$

LoRA 将 ΔW 分解为两个小矩阵的乘积：

$$W_{\text{updated}} = W + A \cdot B$$

其中 $A \in \mathbb{R}^{d \times r}$，$B \in \mathbb{R}^{r \times d}$，$r \ll d$。

利用矩阵乘法的分配律，推理时可以保持权重分离：

$$x \cdot W_{\text{updated}} = x \cdot (W + A \cdot B) = x \cdot W + x \cdot A \cdot B$$

**LoRA 的关键优势**：

| 优势 | 说明 |
|------|------|
| 参数极少 | 只训练 A 和 B，参数量 < 2% 原始参数 |
| 原始权重冻结 | 不修改预训练权重，保留通用能力 |
| 热插拔 | 训练好的 LoRA adapter 可随时加载/卸载 |
| 零推理开销 | 训练后可将 A·B 合并到 W 中，无额外延迟 |

---

### 2.2 数据集准备（Section E.2）

复用第 6 章的 SMS Spam Collection 数据集——一个包含约 5,574 条短信的垃圾分类数据集（`ham` 正常 / `spam` 垃圾）。

```python
import pandas as pd
from previous_chapters import (
    download_and_unzip_spam_data,
    create_balanced_dataset,
    random_split
)

# 下载数据
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

# 处理与划分
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
balanced_df = create_balanced_dataset(df)
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
```

数据划分比例：70% 训练 / 10% 验证 / 20% 测试（与第 6 章完全一致）。

> 数据集的详细处理流程见 [07-分类微调.md](07-分类微调.md)。

---

### 2.3 模型初始化（Section E.3）

加载 GPT-2 small（124M）预训练权重，并替换输出头用于 2 分类：

```python
from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt

CHOOSE_MODEL = "gpt2-small (124M)"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
}

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)

# 替换输出头用于 2 分类
num_classes = 2
model.out_head = torch.nn.Linear(in_features=768, out_features=num_classes)
```

**验证模型加载正确**：用 `generate_text_simple` 生成文本，确认输出连贯。

**未微调时的初始准确率**（约 50%，即随机猜测）：

```
Training accuracy:   46.25%
Validation accuracy: 45.00%
Test accuracy:       48.75%
```

---

### 2.4 LoRA 核心实现（Section E.4）⭐

这是本附录的核心部分。LoRA 的实现由三个类/函数组成，层层递进：

#### LoRALayer —— 低秩分解层

```python
import math

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # A: 非零初始化
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))  # B: 零初始化！
        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        x = (self.alpha / self.rank) * (x @ self.A @ self.B)
        return x
```

**逐行解析**：

| 代码 | 说明 |
|------|------|
| `self.A = Parameter(empty(in_dim, rank))` | A 矩阵形状 `(d, r)`，例如 `(768, 16)` |
| `kaiming_uniform_(self.A)` | 使用 Kaiming 初始化 A，与标准 Linear 层一致 |
| `self.B = Parameter(zeros(rank, out_dim))` | B 矩阵形状 `(r, d)`，初始化为 **全零** |
| `(self.alpha / self.rank) * (x @ self.A @ self.B)` | 计算 LoRA 贡献，带缩放因子 |

**为什么 B 初始化为零？**

这是 LoRA 设计的关键：$B = 0$ 意味着 $A \cdot B = 0$，因此 LoRA 的初始贡献为零。模型在训练开始时的行为完全等同于原始预训练模型——不会破坏已学到的知识。训练过程中 B 逐渐学到非零值，LoRA 的贡献逐步增大。

**alpha/rank 缩放因子的作用**：

`alpha / rank` 这个比例使得不同 rank 设置下 LoRA 的贡献量级保持一致，从而在切换 rank 时无需重新调整学习率。这是 LoRA 实践中的一个重要技巧。

---

#### LinearWithLoRA —— 组合层

```python
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear           # 原始冻结的 Linear 层
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)   # xW + xAB = x(W + AB)
```

**工作原理**：

- `self.linear`：保留原始的 `nn.Linear` 层（权重已冻结，`requires_grad=False`）
- `self.lora`：新增的 LoRA 层（A 和 B 可训练，`requires_grad=True`）
- `forward`：将两条路径的结果相加，实现 $x \cdot W + x \cdot A \cdot B$

这正好对应了 LoRA 的数学原理：$x \cdot (W + A \cdot B) = x \cdot W + x \cdot A \cdot B$。

---

#### replace_linear_with_lora —— 递归替换

```python
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # 将 Linear 替换为 LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 递归处理子模块
            replace_linear_with_lora(module, rank, alpha)
```

**工作原理**：

1. 遍历模型的所有直接子模块（`named_children`）
2. 如果子模块是 `nn.Linear`，用 `LinearWithLoRA` 包装替换
3. 如果不是（如 `TransformerBlock`、`Sequential`），递归进入其子模块

**哪些层被替换了？**

GPT-2 small 有 12 个 TransformerBlock，每个 block 包含 6 个 Linear 层：

| 位置 | 层名 | 形状 |
|------|------|------|
| MultiHeadAttention | W_query | (768, 768) |
| MultiHeadAttention | W_key | (768, 768) |
| MultiHeadAttention | W_value | (768, 768) |
| MultiHeadAttention | out_proj | (768, 768) |
| FeedForward | layers[0] | (768, 3072) |
| FeedForward | layers[2] | (3072, 768) |

每个 block 6 个 × 12 个 block + 1 个 out_head = **73 个 Linear 层**全部被替换。

---

#### 应用 LoRA 的完整流程

```python
# 步骤 1：冻结所有原始参数
for param in model.parameters():
    param.requires_grad = False

# 步骤 2：用 LoRA 替换所有 Linear 层
# 新增的 LoRA 参数（A 和 B）默认 requires_grad=True
replace_linear_with_lora(model, rank=16, alpha=16)

# 验证可训练参数数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable LoRA parameters: {total_params:,}")
# 输出：Total trainable LoRA parameters: 2,666,528
```

替换后的模型结构（以第一个 TransformerBlock 为例）：

```
TransformerBlock(
  (att): MultiHeadAttention(
    (W_query): LinearWithLoRA(
      (linear): Linear(in_features=768, out_features=768, bias=True)
      (lora): LoRALayer()
    )
    (W_key): LinearWithLoRA(...)
    (W_value): LinearWithLoRA(...)
    (out_proj): LinearWithLoRA(...)
  )
  (ff): FeedForward(
    (layers): Sequential(
      (0): LinearWithLoRA(
        (linear): Linear(in_features=768, out_features=3072, bias=True)
        (lora): LoRALayer()
      )
      (1): GELU()
      (2): LinearWithLoRA(
        (linear): Linear(in_features=3072, out_features=768, bias=True)
        (lora): LoRALayer()
      )
    )
  )
  ...
)
```

**验证 LoRA 不影响初始行为**：替换后再次计算准确率，仍然是约 50%（与替换前完全一致），证明 B=0 的初始化确保了 LoRA 的零影响启动。

---

### 2.5 训练与结果

#### 训练配置

```python
import time
from previous_chapters import train_classifier_simple

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.1)

num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)
```

| 超参数 | 值 |
|--------|-----|
| rank | 16 |
| alpha | 16 |
| optimizer | AdamW |
| learning rate | 8e-4 |
| weight_decay | 0.1 |
| epochs | 5 |
| batch_size | 8 |

#### 训练过程日志

```
Ep 1 (Step 000000): Train loss 3.820, Val loss 3.462
Ep 1 (Step 000050): Train loss 0.346, Val loss 0.325
Ep 1 (Step 000100): Train loss 0.063, Val loss 0.144
Training accuracy: 100.00% | Validation accuracy: 92.50%
Ep 2 (Step 000150): Train loss 0.054, Val loss 0.045
Ep 2 (Step 000200): Train loss 0.058, Val loss 0.122
Ep 2 (Step 000250): Train loss 0.041, Val loss 0.199
Training accuracy: 100.00% | Validation accuracy: 95.00%
Ep 3 (Step 000300): Train loss 0.020, Val loss 0.153
Training accuracy: 100.00% | Validation accuracy: 95.00%
Ep 4 (Step 000400): Train loss 0.017, Val loss 0.099
Training accuracy: 97.50% | Validation accuracy: 92.50%
Ep 5 (Step 000550): Train loss 0.019, Val loss 0.252
Training accuracy: 100.00% | Validation accuracy: 100.00%
Training completed in 2.16 minutes.
```

#### 参数量对比

| 指标 | 全参数微调 | LoRA 微调 |
|------|-----------|----------|
| 总参数 | 124,441,346 | 124,441,346 |
| 可训练参数 | 124,441,346 | 2,666,528 |
| 训练比例 | 100% | ~2.1% |
| 缩减倍数 | -- | ~50x |

#### 最终准确率

| 数据集 | LoRA 微调准确率 |
|--------|----------------|
| 训练集 | 99.81% |
| 验证集 | 97.99% |
| 测试集 | 96.67% |

**对比第 6 章全参数微调结果**：LoRA 用仅 2.1% 的可训练参数，达到了与全参数微调相当的准确率（测试集约 96-97%）。训练时间也显著缩短。

#### 不同设备的训练时间

| 设备 | 训练时间 |
|------|---------|
| MacBook M1 (CPU) | 12.10 分钟 |
| MacMini M4 Pro (MPS) | 2.16 分钟 |
| Jetson Nano (CUDA) | 3.50 分钟 |
| DGX Spark (CUDA) | 1.02 分钟 |

---

### 2.6 rank 和 alpha 的选择

#### rank（秩）

rank 控制 LoRA 瓶颈层的维度，直接决定可训练参数数量：

| rank | 可训练参数量（GPT-2） | 适用场景 |
|------|----------------------|---------|
| 4 | ~666K | 简单任务、极少数据 |
| 8 | ~1.33M | 通用任务 |
| **16** | **~2.67M** | **推荐默认值** |
| 32 | ~5.33M | 复杂任务 |
| 64 | ~10.67M | 高度复杂任务、大量数据 |

- **rank 过低**：参数量不足，可能欠拟合，无法充分学习任务特征
- **rank 过高**：参数量接近全参数微调，失去 LoRA 的优势
- **rank=16** 是 GPT-2 规模模型的良好默认值

#### alpha（缩放因子）

alpha 控制 LoRA 贡献相对于原始权重的强度：

- **alpha = rank**（最常见）：如 `alpha=16, rank=16`，缩放比 = 1
- **alpha = 2 × rank**：LoRA 贡献更强，适合任务与预训练分布差异较大的场景
- alpha/rank 的比值将 rank 的选择与学习率解耦

#### 选择策略

```
推荐起点：rank=16, alpha=16, lr=8e-4

如果验证集准确率不够：
  1. 先尝试增加 epoch
  2. 再尝试 rank=32 或 alpha=32
  3. 最后尝试调高学习率

如果出现过拟合：
  1. 降低 rank（如 rank=8）
  2. 增加 weight_decay
  3. 减少 epoch
```

---

## 三、动手练习：跑通附录 E 全流程

### 步骤 1：打开 notebook

打开 `appendix-E/01_main-chapter-code/appendix-E.ipynb`。

### 步骤 2：加载 GPT-2 预训练模型并准备数据集

```python
from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt

# 加载模型
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)

# 替换输出头
num_classes = 2
model.out_head = torch.nn.Linear(in_features=768, out_features=num_classes)
```

### 步骤 3：实现 LoRALayer 和 LinearWithLoRA

```python
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        return (self.alpha / self.rank) * (x @ self.A @ self.B)


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

### 步骤 4：应用 replace_linear_with_lora

```python
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)
```

### 步骤 5：冻结原始参数，训练 LoRA 参数

```python
# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 替换 Linear 层（LoRA 参数自动可训练）
replace_linear_with_lora(model, rank=16, alpha=16)

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.1)
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=5, eval_freq=50, eval_iter=5,
)
```

### 步骤 6：评估并与全参数微调对比

```python
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy:   {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy:       {test_accuracy*100:.2f}%")
```

预期输出：
```
Training accuracy:   99.81%
Validation accuracy: 97.99%
Test accuracy:       96.67%
```

---

## 四、常见问题

<details>
<summary><strong>Q1：LoRA 和全参数微调效果差多少？</strong></summary>

在本附录的垃圾分类任务中，LoRA（2.1% 参数）的测试准确率约 96.67%，与全参数微调的结果相当。在大多数实际场景中，LoRA 能达到全参数微调 95-100% 的效果。差距主要出现在任务与预训练分布差异极大的情况下。
</details>

<details>
<summary><strong>Q2：rank 设多大合适？</strong></summary>

没有万能的最优值，但 rank=16 是广泛使用的默认值。简单任务可以降到 rank=4 或 8，复杂任务可以升到 32 或 64。建议从 rank=16 开始，根据验证集表现调整。注意 alpha 通常设为与 rank 相同的值。
</details>

<details>
<summary><strong>Q3：为什么 B 初始化为零，而不是 A？</strong></summary>

B 初始化为零确保了 $A \cdot B = 0$，使得 LoRA 在训练开始时对模型输出没有影响——模型行为完全等同于原始预训练模型。这是 LoRA 设计的核心：从预训练权重出发，逐步学习任务特定的增量。

A 也可以初始化为零（效果等价），但 Kaiming 初始化 A 是更常见的选择，因为它提供了更好的梯度流。
</details>

<details>
<summary><strong>Q4：LoRA 能用在指令微调（第 7 章）吗？</strong></summary>

完全可以。本附录虽然以垃圾分类为例，但 LoRA 的实现是通用的——`replace_linear_with_lora` 会替换模型中所有的 `nn.Linear` 层。只需将 LoRA 应用到第 7 章的指令微调模型上，冻结原始权重后训练 LoRA 参数即可。HuggingFace 的 PEFT 库在实际项目中就是这样做的。
</details>

<details>
<summary><strong>Q5：LoRA 推理时有额外开销吗？</strong></summary>

训练时有两条路径（原始 Linear + LoRA），存在微小开销。但训练完成后，可以将 $A \cdot B$ 合并到原始权重 $W$ 中：$W_{\text{merged}} = W + A \cdot B$。合并后推理与原始模型完全一致，没有任何额外开销。
</details>

<details>
<summary><strong>Q6：QLoRA 和 LoRA 有什么区别？</strong></summary>

QLoRA（Quantized LoRA）是 LoRA 的扩展：先将预训练权重量化到 4-bit（NormalFloat 格式），再在量化后的模型上应用 LoRA。这进一步降低了显存需求，使得在消费级 GPU 上微调大模型（如 65B 参数的 LLaMA）成为可能。核心思路相同，但 QLoRA 多了量化步骤和分页优化器（paged optimizer）来管理显存。
</details>

---

## 五、考试卡片

<details>
<summary><strong>Q22：LoRA 如何实现参数高效微调？</strong></summary>

冻结原始权重矩阵 W，在旁路添加可训练的低秩矩阵 A、B：

$$W' = W + \Delta W = W + A \cdot B$$

其中 $A \in \mathbb{R}^{d \times r}$，$B \in \mathbb{R}^{r \times d}$，$r \ll d$（通常 r=8~64）。

优势：只训练 A 和 B，参数量极少（<2% 原始参数），显存需求大幅降低。推理时可将 A·B 合并到 W 中，无额外开销。
</details>

<details>
<summary><strong>Q23：LoRA 的三个核心组件是什么？各自的作用？</strong></summary>

1. **LoRALayer** — 低秩分解矩阵 A 和 B。A 使用 Kaiming 初始化（非零），B 初始化为零。forward 计算 `(alpha/rank) * (x @ A @ B)`，即 LoRA 对输出的贡献。

2. **LinearWithLoRA** — 包装原始 Linear 层 + LoRA 层。forward 返回 `linear(x) + lora(x)`，实现 $xW + xAB = x(W + AB)$。原始权重冻结，仅 LoRA 参数可训练。

3. **replace_linear_with_lora** — 递归遍历模型的所有子模块，将每个 `nn.Linear` 替换为 `LinearWithLoRA`。在 GPT-2 small 中共替换 73 个 Linear 层。
</details>

<details>
<summary><strong>Q24：为什么 LoRA 初始化时不影响模型行为？</strong></summary>

B 矩阵初始化为零，因此 $A \cdot B = 0$（零矩阵）。LoRA 的贡献初始为零，模型行为完全等同于原始预训练模型。这保证了：
- 训练起点与预训练模型一致
- 不会破坏已学到的通用知识
- LoRA 只在训练过程中逐步学习增量更新

训练过程中 B 逐渐学习非零值，LoRA 贡献逐步增大。
</details>

<details>
<summary><strong>Q25：LoRA 中 rank 和 alpha 两个超参数各自的含义和推荐设置？</strong></summary>

**rank**（秩）：控制 LoRA 瓶颈层的维度 r。A 的形状为 (d, r)，B 的形状为 (r, d)。rank 越大，可训练参数越多，表达能力越强，但参数效率越低。推荐默认值 rank=16。

**alpha**（缩放因子）：控制 LoRA 贡献相对于原始权重的强度。forward 中实际缩放为 `alpha/rank`。将 alpha 与 rank 解耦使得切换 rank 时无需重新调学习率。推荐 alpha = rank（即 alpha=16）。

**经验法则**：从 rank=16, alpha=16 开始，根据验证集表现调整。
</details>

---

## 六、关键文件速查

| 想看什么 | 打开 |
|---------|------|
| LoRA 完整实现 | `appendix-E/01_main-chapter-code/appendix-E.ipynb` |
| 前序章节代码 | `appendix-E/01_main-chapter-code/previous_chapters.py` |
| GPT 权重下载 | `appendix-E/01_main-chapter-code/gpt_download.py` |
| 分类微调基础 | `ch06/01_main-chapter-code/ch06.ipynb` |
| 分类微调学习笔记 | [07-分类微调.md](07-分类微调.md) |

---

## 七、补充资源

附录 E 没有配套视频或额外练习文件。以下是延伸学习的推荐资源：

**论文**：
- LoRA 原论文：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)（Hu et al., 2021）
- QLoRA：[QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314)（Dettmers et al., 2023）

**实战工具**：
- [HuggingFace PEFT](https://github.com/huggingface/peft) — 生产级 LoRA 实现，支持多种模型架构
- 第 6 章 bonus 实验（`ch06/02_bonus_additional-experiments/`）包含 LoRA 与全参数微调的对比结果

**进阶方向**：
- 尝试将 LoRA 应用到第 7 章的指令微调流程
- 阅读 Ch4 bonus 中的高级注意力机制（GQA、MLA、SWA、KV Cache）
- 尝试 Ch5 的 LLM from scratch（Llama、Qwen、Gemma）

---

## 上一篇

第 09 篇：PyTorch 入门 → [09-PyTorch入门.md](09-PyTorch入门.md)

## 下一篇

恭喜！你已完成全书 7 章 + 2 个核心附录的学习。接下来可以：
- 深入 Ch4 bonus 高级注意力机制（GQA、MLA、SWA、KV Cache）
- 尝试 Ch5 的 LLM from scratch（Llama、Qwen、Gemma）
- 在自己的数据集上用 LoRA 进行微调实践
- 阅读附录 D（训练循环进阶技巧：LR scheduler、gradient clipping）
