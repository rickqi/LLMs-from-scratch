# 滑动窗口注意力（Sliding Window Attention, SWA）

本奖励材料说明了使用滑动窗口注意力（SWA）替代常规多头注意力（MHA）时的内存节省。

&nbsp;
## 介绍

什么是滑动窗口注意力（SWA）？如果我们认为常规的自注意力是*全局*注意力机制，因为每个序列元素都可以访问每个其他序列元素，那么我们可以将 SWA 视为*局部*注意力，因为这里我们限制当前查询位置周围的上下文大小。这在下图中有说明。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/1.webp?2" alt="Sliding Window Attention" width="500px" />

如上图所示，每个标记只关注其位置周围的固定大小局部窗口，而不是关注所有之前的标记。这种局部注意力显著降低了 KV 缓存的大小。

在本介绍的其余部分，我们将在 [Gemma 3](https://arxiv.org/abs/2503.19786) 的背景下讨论 SWA，它在 [../../ch05/12_gemma3](../../ch05/12_gemma3) 中从零实现。

滑动窗口注意力最初是在 2020 年的 [LongFormer 论文](https://arxiv.org/abs/2004.05150) 中引入的，但我们关注谷歌的 Gemma 模型是因为它们是非常好的开源模型，表明滑动窗口注意力在最近的、能够实现的模型中确实是一种可行的方法。

[Gemma 2](https://arxiv.org/abs/2408.00118) 使用了结合了局部（滑动窗口）和全局注意力层的混合方法，比例为 1:1。每个标记可以访问 4k 标记的上下文窗口。这种 1:1 混合的原因是它在效率和全局上下文建模之间取得了平衡，因为仅使用局部注意力的 LLM 可能会过于受限。

[Gemma 3](https://arxiv.org/abs/2503.19786) 进一步朝着效率方向发展。它在滑动窗口和全注意力层之间使用了 5:1 的比例，这意味着每五个局部注意力层就有一个全局层。此外，滑动窗口大小从 Gemma 2 中的 4096 标记减少到 Gemma 3 中的 1024 标记。

有趣的是，Gemma 3 技术报告中的消融研究表明，这些变化对整体模型质量只有很小的影响。换句话说，通过滑动窗口注意力实现的显著内存和计算节省只带来了建模性能的微小损失。


&nbsp;
## 滑动窗口注意力（SWA）内存节省

内存节省主要反映在 KV 存储中。我们可以使用以下公式计算 KV 存储大小：

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

使用 SWA 时，我们将上面的序列长度（seqlen）替换为窗口大小 W。因此，当使用滑动窗口注意力时，我们将 KV 缓存大小按"W / seqlen"的比例减少。（注意，为简单起见，这假设在每个层都使用滑动窗口注意力。）

您可以使用此文件夹中的 [memory_estimator_swa.py](memory_estimator_swa.py) 脚本为不同的模型配置应用此公式，以查看通过使用 SWA 而不是 MHA 可以节省多少内存：

```bash
➜ uv run memory_estimator_swa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 1024 --swa_ratio "5:1"
==== Config ====
context_length         : 32768
sliding_window_size    : 1024
emb_dim                : 4096
n_heads                : 32
n_layers               : 32
n_kv_groups            : 4
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 128
GQA n_kv_heads         : 8
Effective SWA window W : 1024
Layer ratio (SWA:Full) : 5:1
Distributed layers     : 27 SWA, 5 FULL

==== KV-cache totals across all layers ====
MHA KV total           : 17.18 GB
GQA KV total           : 4.29 GB
MHA + SWA (Ratio: 5:1) : 3.14 GB
GQA + SWA (Ratio: 5:1) : 0.78 GB
```

请注意，Gemma 3 将 SWA 与 GQA 结合使用。

下图进一步显示了对于不同的上下文长度，使用 SWA 而不是 MHA 的节省情况：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/4.webp?2" alt="SWA" width="800px" />

&nbsp;

您可以通过以下方式重现这些图：

```bash
uv run plot_memory_estimates_swa.py \
  --emb_dim 4096 --n_heads 48 --n_layers 36 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 2048 --swa_ratio "5:1"
```


&nbsp;
## SWA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_swa.py](gpt_with_kv_swa.py) 脚本提供了在 GPT 模型实现的上下文中比较 MHA 和 SWA 内存使用的实际示例。

请注意，SWA 也可以与 MLA 和 GQA 结合使用（如前所述），但为简单起见，这里没有这样做。

请注意，该模型没有经过训练，因此生成无意义的文本。但是，您可以将其用作第 5-7 章中标准 GPT 模型的即用型替换，并进行训练。

此外，此实现使用了 [另一奖励部分](../03_kv-cache) 中解释的 KV 缓存，因此内存节省更加明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_swa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--sliding_window_size 1024 \
--sliding_window_stride 5   # like Gemma 3

...

Time: 514.38 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

我们没有像上图看到的那样节省这么多的原因有两方面：

1. 我使用较小的配置，让模型在合理时间内完成生成。
2. 更重要的是，我们在这里查看整个模型，而不仅仅是注意力机制；模型中的全连接层占用大部分内存（但这是一个单独分析的主题）。