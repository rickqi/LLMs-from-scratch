# 用于线性注意力的门控 DeltaNet

最近，[Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) 和 [Kimi Linear](https://arxiv.org/abs/2510.26692) 提出了混合 transformer，它们实现了对注意力机制的替代方案，这些机制相对于上下文长度呈线性扩展而不是二次扩展。

Qwen3-Next 和 Kimi Linear 都使用 3:1 比率，这意味着每三个采用线性 Gated DeltaNet 变体的 transformer 块，就有一个块使用完整注意力，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/01.webp" alt="Qwen3-Next versus Kimi Linear">

&nbsp;
## 介绍和概述

Gated DeltaNet 是一种线性注意力变体，从循环神经网络中获得灵感，包括来自 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) 论文的门控机制。从某种意义上说，Gated DeltaNet 是带有 Mamba 风格门控的 DeltaNet，而 DeltaNet 是一种线性注意力机制。

Kimi Linear 通过 Kimi Delta Attention (KDA) 机制修改了 Qwen3-Next 的线性注意力机制，这本质上是对 Gated DeltaNet 的改进。而 Qwen3-Next 对每个注意力头应用标量门控来控制内存衰减率，Kimi Linear 用每个特征维度的通道级门控替换了它。根据作者的说法，这提供了对内存的更多控制，从而改善了长上下文推理。

此外，对于完整注意力层，Kimi Linear 将 Qwen3-Next 的门控注意力层（本质上是带有输出门控的标准多头注意力层）替换为多头潜注意力（MLA）。这是我们之前在 DeepSeek V3/R1 部分讨论过的相同 MLA 机制，但带有额外的门控。（回顾一下，MLA 压缩键/值空间以减少 KV 缓存大小。）

Kimi Linear 中的 MLA 不使用门控，这是故意的，这样作者可以更直接地与标准 MLA 比较架构，然而他们 [ stated](https://x.com/yzhang_cs/status/1984631714464088563) 计划在未来添加它。

由于我们已经在 [../05_mla](../05_mla) 中实现了 MLA，此奖励材料专注于 Gated DeltaNet 方面。


&nbsp;
## 门控注意力

在我们进入 Gated DeltaNet 本身之前，让我们简要讨论一下门控。如前图中的 Qwen3-Next 架构上部分所示，Qwen3-Next 使用"门控注意力"（gated attention）。这本质上是在附加的 sigmoid 门控之上的标准完整注意力。

这种门控是一个简单的修改，我将其添加到第 3 章的 `MultiHeadAttention` 代码中，用于说明目的：

```python
import torch
from torch import nn

class GatedMultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### NEW: Add gate
        self.W_gate = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        ####################################################
        ### NEW: Add gate
        gate = self.W_gate(x)
        ####################################################
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(
            mask_bool, torch.finfo(attn_scores.dtype).min
        )

        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)

        ####################################################
        ### NEW: Add gate
        context = context * torch.sigmoid(gate)
        ####################################################
        out = self.out_proj(context)
        return out
```


正如我们在上面看到的，在像往常一样计算注意力之后，模型使用来自相同输入的单独门控信号，应用 sigmoid 将其保持在 0 和 1 之间，并将其与注意力输出相乘。这允许模型动态放大或缩小某些特征。Qwen3-Next 开发者 [ state](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) 门控注意力输出机制有助于消除注意力沉没（Attention Sink）和大规模激活（Massive Activation）等问题，确保模型中的数值稳定性：

> [...] 注意力输出门控机制有助于消除注意力沉没和大规模激活等问题，确保模型中的数值稳定性。


&nbsp;
## 门控 DeltaNet

那么，什么是 Gated DeltaNet？Gated DeltaNet（简称*门控 Delta 网络*）是 Qwen3-Next 的线性注意力层，旨在作为标准 softmax 注意力的替代方案。它如前所述从 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) 论文采用。

Gated DeltaNet 最初被提议为 Mamba2 的改进版本，它将 Mamba2 的门控衰减机制与 delta 规则结合使用。

Mamba 是状态空间模型（transformers 的替代方案），是一个值得在未来单独覆盖的大主题。

delta 规则部分指的是计算新值和预测值之间的差异（delta, Δ）来更新用作内存状态的隐藏状态（稍后详细介绍）。

（顺便说一下，具有经典机器学习文献的读者可以将这看作类似于受生物学启发的 Hebbian 学习："一起发射的神经元连接在一起。"它基本上是感知器更新规则和基于梯度下降的学习的前身，但没有监督。）

Gated DeltaNet 具有类似于上面讨论的门控注意力中的门控，但它使用 SiLU 而不是逻辑 sigmoid 激活，如下图所示。（选择 SiLU 很可能是为了改善梯度流和稳定性，而不是标准 sigmoid。）


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/02.webp" alt="Gated DeltaNet" width=500px>

然而，如上图所示，"gated"在 Gated DeltaNet 中还指几个额外的门控：

- `α`（衰减门控）控制内存随时间衰减或重置的速度，
- `β`（更新门控）控制新输入修改状态的强度。

在代码中，上图所示 Gated DeltaNet 的简化版本（没有卷积混合）可以实现如下（代码受到 Qwen3 团队 [官方实现](https://github.com/huggingface/transformers/blob/0ed6d51ae8ed3f4fafca67a983b8d75bc76cd51b/src/transformers/models/qwen3_next/modular_qwen3_next.py#L835) 的启发）。

（注意，一些实现将衰减门控称为 `gk`（步骤 k 的门控），其中 `exp(gk)` 匹配论文中的 $\alpha_t$。为了保持这种关系明确，下面的代码片段将 log 空间门控 `alpha_log` 与指数衰减 `alpha` 分开。）


```python
import torch
from torch import nn
import torch.nn.functional as F

def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

class GatedDeltaNet(nn.Module):
    def __init__(
        self, d_in, d_out, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### NEW: Gates for delta rule and output gating
        self.W_gate = nn.Linear(d_in, d_out, bias=False)
        self.W_beta = nn.Linear(d_in, d_out, bias=False)

        # Note: The decay gate alpha corresponds to
        # A_log + W_alpha(x) + dt_bias
        self.W_alpha = nn.Linear(d_in, num_heads, bias=False)
        self.dt_bias = torch.Parameter(torch.ones(num_heads))
        A_init = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))
        # We could implement this as
        # W_alpha = nn.Linear(d_in, num_heads, bias=True)
        # but the bias is separate for interpretability and
        # to mimic the official implementation

        self.norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        ####################################################

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        ####################################################
        ### NEW: Compute delta rule gates
        beta = torch.sigmoid(self.W_beta(x))
        alpha_log = -self.A_log.exp().view(1, 1, -1) * F.softplus(
            self.W_alpha(x) + self.dt_bias
        )
        alpha = alpha_log.exp()
        gate = self.W_gate(x)
        ####################################################

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        beta = beta.view(b, num_tokens, self.num_heads, self.head_dim)
        gate = gate.view(b, num_tokens, self.num_heads, self.head_dim)  # NEW

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        beta = beta.transpose(1, 2)

        ####################################################
        ### NEW: QKNorm-like normalization for delta rule
        queries = l2norm(queries, dim=-1) / (self.head_dim ** 0.5)
        keys = l2norm(keys, dim=-1)
        ####################################################

        S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)

        outs = []
        ####################################################
        ### NEW: Gated delta rule update
        for t in range(num_tokens):
            k_t = keys[:, :, t]
            q_t = queries[:, :, t]
            v_t = values[:, :, t]
            b_t = beta[:, :, t]
            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

            S = S * a_t
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * b_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
            ####################################################
            outs.append(y_t)

        context = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
        context = context.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        ### NEW: Apply RMSNorm and SiLU gate
        context = self.norm(context)
        context = context * F.silu(gate)
        ####################################################

        context = context.view(b, num_tokens, self.d_out)
        context = self.dropout(context)
        out = self.out_proj(context)
        return out
```

（注意，为简单起见，我省略了 Qwen3-Next 和 Kimi Linear 使用的卷积混合，以使代码更可读并专注于循环方面。）

所以，如上所示，与标准（或门控）注意力相比有很多不同。

在门控注意力中，模型在所有标记之间计算正常注意力（每个标记都关注或查看每个其他标记）。然后，在获得注意力输出后，门控（一个 sigmoid）决定保留多少该输出。要点是它仍然是与上下文长度呈二次缩放的常规缩点积注意力。

提醒一下，缩点积注意力计算为 softmax(QKᵀ)V，其中 Q 和 K 是 *n* × *d* 矩阵，其中 *n* 是输入标记的数量，*d* 是嵌入维度。所以 QKᵀ 产生一个注意力 *n* × *n* 矩阵，乘以一个 *n* × *d* 维值矩阵 V：

```
attn_scores = queries @ keys.transpose(2, 3)

mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(
    mask_bool, torch.finfo(attn_scores.dtype).min
)

attn_weights = torch.softmax(
    attn_scores / (self.head_dim ** 0.5), dim=-1
)

context = (attn_weights @ values).transpose(1, 2)
context = context.reshape(b, num_tokens, self.d_out)
```


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/03.webp" alt="Quadratic attention" width=500px />

在 Gated DeltaNet 中，没有 *n* × *n* 注意力矩阵。相反，模型一次一个标记地处理标记。它保持一个运行内存（状态），随着每个新标记进入而更新。这就是在代码中实现的，其中 `S` 是为每个时间步 *t* 循环更新的状态。

```python
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
outs = []

for t in range(num_tokens):
    k_t = keys[:, :, t]
    q_t = queries[:, :, t]
    v_t = values[:, :, t]
    b_t = beta[:, :, t]
    a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

    S = S * a_t
    kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * b_t
    S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
```

门控控制内存如何变化：

- α (`alpha`) 调节旧记忆被遗忘（衰减）的程度。

- β (`beta`) 调制当前时间步 *t* 的标记更新记忆的程度。

（并且最终的输出门控，如上代码片段中未显示的，类似于门控注意力；它控制保留多少输出。）

因此，从某种意义上说，Gated DeltaNet 中的这种状态更新类似于循环神经网络（RNN）的工作方式。优点是它通过 for 循环线性缩放，而不是与上下文长度二次缩放。

这种循环状态更新的缺点是，与常规（或门控）注意力相比，它牺牲了来自完整成对注意力的全局上下文建模能力。

Gated DeltaNet 在某种程度上仍然可以捕捉上下文，但它必须通过记忆（*S*）瓶颈。那个记忆是固定大小的，因此更高效，但它将过去的上下文压缩成类似于 RNN 的单个隐藏状态。

这就是为什么 Qwen3-Next 和 Kimi Linear 架构没有用 DeltaNet 层替换所有注意力层，而是使用前面提到的 3:1 比率的原因。

&nbsp;
## DeltaNet 内存节省

在上一节中，我们讨论了 DeltaNet 相对于完整注意力在计算复杂度方面的优势，相对于上下文长度呈线性而不是二次。

除了线性计算复杂度，DeltaNet 的另一个大优势是内存节省，因为 DeltaNet 模块不增长 KV 缓存。（有关 KV 缓存的更多信息，请参见 [../03_kv-cache](../03_kv-cache)）。相反，如前所述，它们保持固定大小的循环状态，因此内存随上下文长度保持恒定。

对于常规多头注意力（MHA）层，我们可以按如下方式计算 KV 缓存大小：

```
KV_cache_MHA ≈ batch_size × n_tokens × n_heads × d_head × 2 × bytes
```

（乘数 2 是因为我们缓存中存储键和值。）

对于我们上面实现的简化 DeltaNet 版本，我们有：

```
KV_cache_DeltaNet = batch_size × n_heads × d_head × d_head × bytes
```

注意 `KV_cache_DeltaNet` 内存大小没有上下文长度（`n_tokens`）依赖性。我们只有存储的内存状态 S，而不是单独的键和值，因此 `2 × bytes` 变为只是 `bytes`。然而，请注意我们现在有二次的 `d_head × d_head`。这来自状态：

```
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
```

但这通常不用担心，因为头维度通常相对较小。例如，在 Qwen3-Next 中它是 128。

带有卷积混合的完整版本更复杂，包括核大小等，但上面的公式应该说明了 Gated DeltaNet 背后的主要趋势和动机。

我们可以通过以下辅助脚本可视化不同上下文长度的内存估计和节省：

```bash
uv run plot_memory_estimates_gated_deltanet.py \
  --emb_dim 2048 \
  --n_heads 16 \
  --n_layers 48 \
  --dtype "bf16"
```

注意，上面计算 `head_dim` 为 `emb_dim / n_heads`。即 2048 / 16 = 128。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/plot.webp" alt="Gated DeltaNet scaling" width=500px>