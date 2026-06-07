# 分组查询注意力（Grouped-Query Attention, GQA）

本奖励材料说明了使用分组查询注意力（GQA）替代常规多头注意力（Multi-Head Attention, MHA）时的内存节省。

&nbsp;
## 介绍

分组查询注意力（GQA）已成为近年来多头注意力的新标准替代方案，它是一种更计算和参数高效的替代方案。请注意，它并不新颖，可以追溯到 2023 年的 [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)。即使是传统的 Llama 2 系列中的较大变体也使用了它。

以下是 GQA 的简要总结。与 MHA 不同，在 MHA 中每个头都有自己的键和值集，为了减少内存使用，GQA 将多个头分组以共享相同的键和值投影。

例如，如下图进一步说明，如果有 3 个键值组和 6 个注意力头，那么第 1 和第 2 个头共享一组键和值，而第 3 和第 4 个头，以及第 5 和第 6 个头分别共享另一组。

&nbsp;

![GQA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/1.webp?1)

&nbsp;

这种键和值的共享减少了键和值计算的总数，从而降低了内存使用并提高了效率。

因此，总结来说，GQA 背后的核心思想是通过在多个查询头之间共享键和值来减少键和值的数量。这（1）降低了模型的参数计数，并且（2）在推理过程中减少了键和值张量的内存带宽使用，因为需要从 KV 缓存中存储和检索的键和值更少。

虽然 GQA 主要是对 MHA 的计算效率的变通方案，但消融研究（如 [原始 GQA 论文](https://arxiv.org/abs/2305.13245) 和 [Llama 2 论文](https://arxiv.org/abs/2307.09288) 中的研究）显示，它在 LLM 建模性能方面与标准 MHA 相当。

然而，这假设键值组的数量经过仔细选择。在所有注意力头共享单个键值组的极端情况下，称为多查询注意力（multi-query attention），内存使用减少得更多，但建模性能可能会受到影响。（而且，在另一个极端，如果我们设置键值组的数量等于查询头的数量，我们又回到了标准的多头注意力。）

&nbsp;
## GQA 内存节省

内存节省主要反映在 KV 存储中。我们可以使用以下公式计算 KV 存储大小：

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

您可以使用此文件夹中的 [memory_estimator_gqa.py](memory_estimator_gqa.py) 脚本为不同的模型配置应用此公式，以查看通过使用 GQA 而不是 MHA 可以节省多少内存：

```bash
➜ uv run memory_estimator_gqa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16
==== Config ====
context_length   : 32768
emb_dim          : 4096
n_heads          : 32
n_layers         : 32
n_kv_groups      : 4
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 128
GQA n_kv_heads   : 8

==== KV-cache totals across all layers ====
MHA total KV cache  : 17.18 GB
GQA total KV cache  : 4.29 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
```

下图进一步显示了在不同上下文长度下使用 GQA 而不是 MHA 的节省情况：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/3.webp?4" alt="GQA" width="500px" />

&nbsp;

您可以通过 `uv run plot_memory_estimates_gqa.py` 重现该图。

&nbsp;
## GQA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_gqa.py](gpt_with_kv_gqa.py) 脚本提供了在 GPT 模型实现的上下文中比较 MHA 和 GQA 内存使用的实际示例。

请注意，GQA 也用于 [Llama 3](../../ch05/07_gpt_to_llama)、[Gemma 3](../../ch05/12_gemma3) 和 [Qwen3](../../ch05/11_qwen3) 奖励材料中。但是，为了简单起见，此文件夹中的代码脚本修改了 GPT 架构，该架构传统上不使用 GQA。

请注意，该模型没有经过训练，因此生成无意义的文本。但是，您可以将其用作第 5-7 章中标准 GPT 模型的即用型替换，并进行训练。

此外，此实现使用了 [另一奖励部分](../03_kv-cache) 中解释的 KV 缓存，因此内存节省更加明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_gqa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--n_kv_groups 4

...

Time: 516.33 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

我们没有像上图看到的那样节省这么多的原因有两方面：

1. 我使用较小的配置，让模型在合理时间内完成生成。
2. 更重要的是，我们在这里查看整个模型，而不仅仅是注意力机制；模型中的全连接层占用大部分内存（但这是一个单独分析的主题）。