# PyTorch Buffers 深入理解：让 GPU 设备迁移不再出错

> 🎬 基于 Sebastian Raschka《Build a Large Language Model (From Scratch)》配套视频
> 📺 [原视频](https://www.youtube.com/watch?v=PetlIokI9Ao) | ⏱️ 约 13 分钟

---

## 这个视频讲什么？

本视频通过一个实际的因果注意力（Causal Attention）模块，演示了 PyTorch 中 **buffer** 与普通 tensor 的关键区别：当你把模型从 CPU 转移到 GPU 时，普通 tensor 不会自动跟随迁移，而通过 `register_buffer` 注册的 tensor 会自动随模型一起迁移。视频还补充了 buffer 在 `state_dict` 保存/加载中的优势。

---

## 一、为什么需要 Buffer？先看一个报错

### 1.1 不使用 Buffer 的因果注意力模块

以下代码来自第 3 章的因果注意力实现，其中 `mask`（因果掩码）直接赋值为一个普通 `torch.Tensor`：

```python
import torch
import torch.nn as nn

class CausalAttentionWithoutBuffers(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # mask 是一个普通 tensor，不是 parameter，也不是 buffer
        self.mask = torch.triu(
            torch.ones(context_length, context_length), diagonal=1
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
```

### 1.2 CPU 上运行没有问题

```python
torch.manual_seed(123)

inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your     (x^1)
    [0.55, 0.87, 0.66],  # journey  (x^2)
    [0.57, 0.85, 0.64],  # starts   (x^3)
    [0.22, 0.58, 0.33],  # with     (x^4)
    [0.77, 0.25, 0.10],  # one      (x^5)
    [0.05, 0.80, 0.55],  # step     (x^6)
])

batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]
d_in = inputs.shape[1]
d_out = 2

ca_without_buffer = CausalAttentionWithoutBuffers(d_in, d_out, context_length, 0.0)

with torch.no_grad():
    context_vecs = ca_without_buffer(batch)

print(context_vecs)
# 输出正常，没有问题
```

### 1.3 迁移到 GPU 后报错

当把模型转移到 GPU 时：

```python
# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 转移数据到 GPU（tensor 需要重新赋值）
batch = batch.to(device)

# 转移模型到 GPU（module 不需要重新赋值）
ca_without_buffer = ca_without_buffer.to(device)

# 再次运行
with torch.no_grad():
    context_vecs = ca_without_buffer(batch)
```

报错信息：

```
RuntimeError: expected self and mask to be on the same device,
but got mask on cpu and self on cuda:0
```

> 意思是：`attn_scores`（在 GPU 上）和 `self.mask`（还在 CPU 上）不在同一个设备上。

### 1.4 问题诊断

```python
print("W_query.device:", ca_without_buffer.W_query.weight.device)
# W_query.device: cuda:0    <-- weight 是 Parameter，自动迁移了

print("mask.device:", ca_without_buffer.mask.device)
# mask.device: cpu           <-- mask 是普通 Tensor，没有迁移！

print(type(ca_without_buffer.mask))
# torch.Tensor               <-- 不是 nn.Parameter
```

**根本原因**：PyTorch 的 `module.to(device)` 只会自动迁移两种东西：

| 类型 | 自动迁移 | 示例 |
|------|---------|------|
| `nn.Parameter` | 是 | `W_query.weight` |
| 注册的 Buffer | 是 | `register_buffer("mask", ...)` |
| 普通 `torch.Tensor` | **否** | `self.mask = torch.triu(...)` |

---

## 二、Buffer 是什么？核心概念

### 2.1 一句话定义

> **Buffer** 是注册在 `nn.Module` 中的 tensor，它**不需要梯度更新**（不像 Parameter），但会**随模型自动迁移设备**并**出现在 `state_dict` 中**。

### 2.2 Parameter vs Buffer vs 普通 Tensor

```
nn.Module 内部的三种 tensor：

┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  nn.Parameter          register_buffer         普通 tensor       │
│  ┌────────────┐       ┌────────────┐        ┌────────────┐     │
│  │ 需要梯度    │       │ 不需要梯度  │        │ 不管梯度    │     │
│  │ 优化器更新  │       │ 不参与训练  │        │ 不参与训练  │     │
│  │ 自动迁设备  │       │ 自动迁设备  │        │ 不自动迁移  │     │
│  │ 在state_dict│      │ 在state_dict│       │ 不在dict中  │     │
│  └────────────┘       └────────────┘        └────────────┘     │
│                                                                  │
│  权重矩阵 W_query      因果掩码 mask         临时中间变量        │
│  偏置项 bias           dropout mask          计数器等            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.3 为什么 mask 不应该是 Parameter？

从技术上讲，你可以把 mask 包裹成 `nn.Parameter`，让它在设备迁移时自动跟随。但这有两个问题：

1. **语义错误**：Parameter 的含义是"需要通过训练学习的参数"。mask 是固定的上三角矩阵，不应被优化器更新
2. **副作用**：Parameter 会被传递给优化器，白白浪费计算资源

因此，**Buffer 才是正确的选择**：不需要梯度，但需要跟随模型迁移设备。

---

## 三、使用 register_buffer 修复问题

### 3.1 修改后的代码

只需改动一行：

```python
class CausalAttentionWithBuffer(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # 旧写法（不自动迁移设备）:
        # self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

        # 新写法（使用 register_buffer）:
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # forward 方法完全不变
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
```

### 3.2 register_buffer 语法

```python
self.register_buffer(name: str, tensor: torch.Tensor, persistent: bool = True)
```

| 参数 | 含义 |
|------|------|
| `name` | 字符串，注册后可以通过 `self.name` 访问 |
| `tensor` | 要注册的 tensor |
| `persistent` | 是否包含在 `state_dict` 中（默认 True） |

### 3.3 验证修复效果

```python
ca_with_buffer = CausalAttentionWithBuffer(d_in, d_out, context_length, 0.0)
ca_with_buffer.to(device)

# 检查设备
print("W_query.device:", ca_with_buffer.W_query.weight.device)
# W_query.device: cuda:0

print("mask.device:", ca_with_buffer.mask.device)
# mask.device: cuda:0    <-- 现在也自动迁移了！

# 正常运行
with torch.no_grad():
    context_vecs = ca_with_buffer(batch)
print(context_vecs)
# 输出正常，不再报错
```

### 3.4 来回切换设备也没问题

Buffer 注册后，模型在 CPU 和 GPU 之间来回切换时，mask 会自动跟随：

```python
# 转到 GPU
ca_with_buffer.to("cuda")
print(ca_with_buffer.mask.device)  # cuda:0

# 转回 CPU
ca_with_buffer.to("cpu")
print(ca_with_buffer.mask.device)  # cpu
```

---

## 四、Buffer 与 state_dict：保存/加载模型

Buffer 的另一个重要优势是会被包含在模型的 `state_dict` 中，使得保存和加载模型时 buffer 的状态也会被持久化。

### 4.1 没有 Buffer 的 state_dict

```python
ca_without_buffer.state_dict()
# 只包含 Parameter，mask 不在其中：
# OrderedDict([
#   ('W_query.weight', tensor(...)),
#   ('W_key.weight',   tensor(...)),
#   ('W_value.weight', tensor(...))
# ])
```

### 4.2 有 Buffer 的 state_dict

```python
ca_with_buffer.state_dict()
# mask 也被包含进来：
# OrderedDict([
#   ('mask',           tensor([[0., 1., 1., 1., 1., 1.],
#                              [0., 0., 1., 1., 1., 1.],
#                              [0., 0., 0., 1., 1., 1.],
#                              [0., 0., 0., 0., 1., 1.],
#                              [0., 0., 0., 0., 0., 1.],
#                              [0., 0., 0., 0., 0., 0.]])),
#   ('W_query.weight', tensor(...)),
#   ('W_key.weight',   tensor(...)),
#   ('W_value.weight', tensor(...))
# ])
```

### 4.3 保存/加载演示

假设我们修改了 mask 的值（将 1 改为 2），然后保存并重新加载：

```python
# 修改 buffer 值
ca_with_buffer.mask[ca_with_buffer.mask == 1.] = 2.

# 保存
torch.save(ca_with_buffer.state_dict(), "model.pth")

# 加载到新实例
new_ca = CausalAttentionWithBuffer(d_in, d_out, context_length, 0.0)
new_ca.load_state_dict(torch.load("model.pth"))

print(new_ca.mask)
# tensor([[0., 2., 2., 2., 2., 2.],
#         [0., 0., 2., 2., 2., 2.],
#         ...
# 修改后的值被正确恢复了！
```

而如果不使用 buffer，保存/加载后 mask 会恢复为初始值（1），修改丢失。

---

## 五、因果掩码（Causal Mask）详解

视频中使用的 mask 是一个上三角矩阵，用于实现因果注意力机制：

```
context_length = 6 时，mask 矩阵如下：

     t1  t2  t3  t4  t5  t6
t1 [  0   1   1   1   1   1 ]
t2 [  0   0   1   1   1   1 ]
t3 [  0   0   0   1   1   1 ]
t4 [  0   0   0   0   1   1 ]
t5 [  0   0   0   0   0   1 ]
t6 [  0   0   0   0   0   0 ]

0 = 可以看到（对角线及以下）
1 = 需要屏蔽（上三角部分）

作用：确保 token i 只能看到 token 0 到 token i，
     不能"偷看"未来的 token
```

```python
# 生成因果掩码的代码
mask = torch.triu(torch.ones(6, 6), diagonal=1)
```

### 为什么要预计算 mask？

Sebastian 在视频中提到，将 mask 的创建放在 `__init__` 中而不是 `forward` 中，是出于效率考虑。如果每次前向传播都重新创建 mask，会损失约 **30% 的效率**。

```
效率对比：

__init__ 中创建一次 mask：  ████████████████████  100%
forward 中每次创建 mask：   ██████████████        ~70%
```

---

## 六、tensor.to() 的注意事项

视频中还提到一个 PyTorch 的细节：

```python
# tensor（数据）：必须重新赋值，否则不会生效
batch = batch.to(device)        # 正确
batch.to(device)                # 错误！不会生效

# module（模型）：不需要重新赋值，in-place 操作
ca_with_buffer.to(device)       # 正确
ca_with_buffer = ca_with_buffer.to(device)  # 也可以，但不必
```

---

## 关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| `register_buffer` | 在 Module 中注册一个不需要梯度的 tensor，使其自动跟随设备迁移 |
| Parameter | 需要通过训练学习的参数，会被优化器更新，自动跟随设备迁移 |
| Buffer | 不需要学习的固定 tensor，但需要随模型迁移设备和保存/加载 |
| 普通 Tensor | 既不会自动迁移设备，也不会出现在 `state_dict` 中 |
| 因果掩码 (Causal Mask) | 上三角矩阵，防止注意力机制"看到"未来 token |
| `torch.triu` | 生成上三角矩阵，用于创建因果掩码 |
| `state_dict` | 模型状态的字典，包含所有 Parameter 和 Buffer |
| `masked_fill_` | 将掩码位置的值替换为指定值（如 -inf） |
| `persistent` | register_buffer 的参数，控制是否包含在 state_dict 中 |

## 常见问题

### Q1：什么时候应该用 buffer 而不是普通 tensor？

当你有一个**不需要梯度更新**但**需要随模型一起迁移设备**的 tensor 时，就应该用 `register_buffer`。常见场景包括因果掩码（causal mask）、位置编码（positional encoding，如果不用 nn.Embedding 的话）、dropout mask 等。

### Q2：能不能用 `nn.Parameter` 代替 buffer？

技术上可以，但不推荐。`nn.Parameter` 会被传递给优化器参与训练，对于一个固定的掩码来说这是语义错误且浪费资源。Buffer 的设计初衷就是"不参与训练但要跟随模型"。

### Q3：没有 GPU 还需要用 buffer 吗？

即使没有 GPU，buffer 仍然有用：它会出现在 `state_dict` 中，使得保存和加载模型时 buffer 的值会被正确保存和恢复。不过设备迁移的优势确实主要体现在有 GPU 的场景。

### Q4：`register_buffer` 的 `persistent=False` 是什么意思？

设置 `persistent=False` 后，该 buffer 仍然会随模型自动迁移设备，但**不会出现在 `state_dict` 中**。适用于需要在运行时存在但不需要持久化保存的 tensor。

### Q5：为什么在 `__init__` 而不是 `forward` 中创建 mask？

效率原因。每次前向传播都重新创建 mask 会导致约 30% 的性能损失。在 `__init__` 中创建一次（并注册为 buffer）是最优做法。

## 下一步

本视频是第 3 章"注意力机制"的补充内容。理解了 buffer 的用法后，可以继续学习：

- [下一节：GPT 模型实现跟练](07-GPT模型实现-教程.md) -- 将注意力机制组装成完整的 GPT 架构
- [对应代码笔记](ch03/03_understanding-buffers/understanding-buffers.ipynb) -- 动手运行完整代码
- [上一节：注意力机制教程](05-注意力机制-教程.md) -- 回顾 self-attention 的完整实现

## 关键文件速查

| 想看什么 | 打开 |
|---------|------|
| Buffer 完整代码示例 | `ch03/03_understanding-buffers/understanding-buffers.ipynb` |
| 第 3 章主代码 | `ch03/01_main-chapter-code/ch03.ipynb` |
| 多头注意力实现 | `ch03/01_main-chapter-code/multihead-attention.ipynb` |
| 高效注意力变体对比 | `ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb` |
| 视频字幕原文 | `docs/youtube/06-PyTorch-Buffers.md` |
