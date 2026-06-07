# 多头潜注意力（Multi-Head Latent Attention, MLA）

本奖励材料说明了使用多头潜注意力（MLA）替代常规多头注意力（MHA）时的内存节省。

&nbsp;
## 介绍

在 [../04_gqa](../04_gqa) 中，我们讨论了分组查询注意力（GQA）作为对 MHA 的计算效率变通方案。而消融研究（如 [原始 GQA 论文](https://arxiv.org/abs/2305.13245) 和 [Llama 2 论文](https://arxiv.org/abs/2307.09288) 中的研究）显示，它在 LLM 建模性能方面与标准 MHA 相当。

现在，多头潜注意力（MLA）在 [DeepSeek V2, V3, and R1](https://arxiv.org/abs/2412.19437) 中使用，提供了不同的内存节省策略，这种策略与 KV 缓存尤其配合良好。与 GQA 共享键和值头不同，MLA 在将键和值张量存储在 KV 缓存之前将其压缩到较低维度空间。

在推理时，这些压缩的张量在使用前会被投影回其原始大小，如下图所示。这增加了额外的矩阵乘法但减少了内存使用。

&nbsp;

![MLA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/1.webp)

&nbsp;

（顺便说一下，查询也被压缩了，但仅用于训练，不用于推理。）

顺便一提，如前所述，MLA 在 DeepSeek V3 中并不新鲜，因为它的 [DeepSeek V2 前身](https://arxiv.org/abs/2405.04434) 也使用（甚至引入了）它。此外，V2 论文包含一些有趣的消融研究，可能解释了为什么 DeepSeek 团队选择 MLA 而不是 GQA（见下图）。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/2.webp" alt="GQA" width="500px" />

&nbsp;

如上图所示，GQA 似乎比 MHA 表现更差，而 MLA 在建模性能方面优于 MHA，这很可能是 DeepSeek 团队选择 MLA 而不是 GQA 的原因。（如果 MLA 和 GQA 之间的"每标记 KV 缓存"节省比较也很有趣！）

在继续讨论下一个架构组件之前，总结一下本节，MLA 是一个聪明的技巧，可以在减少 KV 缓存内存使用的同时，甚至在建模性能上略微优于 MHA。

&nbsp;
## MLA 内存节省

内存节省主要反映在 KV 存储中。我们可以使用以下公式计算 KV 存储大小：

bytes ≈ batch_size × seqlen × n_layers × latent_dim × bytes_per_elem

相比之下，MHA KV 缓存内存计算如下：

bytes ≈ batch_size × seqlen × n_layers × embed_dim × 2 (K,V) × bytes_per_elem

这意味着，在 MLA 中，我们将"embed_dim × 2 (K,V)"减少到"latent_dim"，因为我们只存储压缩的潜表示，而不是完整的键和值向量，如上图中较早显示的那样。

您可以使用此文件夹中的 [memory_estimator_mla.py](memory_estimator_mla.py) 脚本为不同的模型配置应用此公式，以查看通过使用 MLA 而不是 MHA 可以节省多少内存：

```bash
➜ uv run memory_estimator_mla.py \
  --context_length 8192 \
  --emb_dim 2048 \
  --n_heads 24 \
  --n_layers 48 \
  --n_kv_groups 4 \
  --batch_size 1 \
  --dtype bf16 \
  --latent_dim 1024
==== Config ====
context_length   : 8192
emb_dim          : 2048
n_heads          : 24
n_layers         : 48
n_kv_groups      : 4
latent_dim       : 1024
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 86
GQA n_kv_heads   : 6

==== KV-cache totals across all layers ====
MHA total KV cache  : 3.25 GB
GQA total KV cache  : 0.81 GB
MLA total KV cache  : 0.81 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
Ratio (MHA / MLA)   : 4.03x
Savings (MLA vs MHA): 75.19%
```

请注意，上面的压缩（`--emb_dim 2048 -> latent_dim 1024`）是为了实现与 GQA 相似的节省。在实践中，压缩是一个需要仔细研究的超参数，因为选择 `latent_dim` 太小可能会对建模性能产生负面影响（类似于在 GQA 中选择太多的 `n_kv_groups`）。

下图进一步显示了对于不同的 `latent_dim` 值，使用 MLA 而不是 MHA 的节省情况，作为上下文长度的函数：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/3.webp?2" alt="GQA" width="500px" />

&nbsp;

您可以通过 `uv run plot_memory_estimates_mla.py` 重现该图。


&nbsp;
## MLA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_mla.py](gpt_with_kv_mla.py) 脚本提供了在 GPT 模型实现的上下文中比较 MHA 和 MLA 内存使用的实际示例。

这里的 MLA 代码受到 [https://huggingface.co/bird-of-paradise/deepseek-mla](https://huggingface.co/bird-of-paradise/deepseek-mla) 实现的启发。

请注意，MLA 也可以与 [GQA](../04_gqa) 结合使用，但为了简单起见，这里没有这样做。（目前，我也不知道有 prominent LLM 这样做。）

还请注意，该模型没有经过训练，因此生成无意义的文本。但是，您可以将其用作第 5-7 章中标准 GPT 模型的即用型替换，并进行训练。

最后，此实现使用了 [另一奖励部分](../03_kv-cache) 中解释的 KV 缓存，因此内存节省更加明显。

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
uv run gpt_with_kv_mla.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--latent_dim 192 # (768×2)/192 = 8× compression

...

Time: 487.21 sec
67 tokens/sec
Max memory allocated: 0.68 GB
```

我们没有像上图看到的那样节省这么多的原因有两方面：

1. 我使用较小的配置，让模型在合理时间内完成生成。
2. 更重要的是，我们在这里查看整个模型，而不仅仅是注意力机制；模型中的全连接层占用大部分内存（但这是一个单独分析的主题）。