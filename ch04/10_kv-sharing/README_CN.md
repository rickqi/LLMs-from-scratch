# 跨层 KV 共享

本奖励材料说明了在使用 KV 缓存的同时使用跨层 KV 共享时的内存节省。

&nbsp;
## 介绍

在 [../04_gqa](../04_gqa) 中，我们讨论了分组查询注意力（GQA），其中多个查询头共享相同的键和值头。跨层 KV 共享将类似的想法应用于 transformer 层之间。

不是在每个层中计算新的键和值投影，后面的层从较早层重用 K/V 张量。它们仍然计算自己的查询，因此每层可以形成自己的注意力模式。主要的内存节省来自于在缓存中存储更少的 K/V 张量。

这个想法也称为跨层注意力。它在 Brandon *et al.* 的 [使用跨层注意力减少 Transformer 键-值缓存大小](https://arxiv.org/abs/2405.12981) 中有描述。Gemma 4 E2B 和 E4B 使用相关的共享 KV 缓存方案，这使其成为本章中 GQA、MLA 和 SWA 示例的有用补充。

<img src="gemma4-kv-sharing.webp" alt="Cross-layer KV sharing" width="800px" />

&nbsp;

在 [Gemma 4](../../ch05/17_gemma4) 中，KV 共享与 GQA 或 MQA 以及滑动窗口注意力结合使用。为了保持此文件夹中简化 GPT 示例的专注，我们只实现跨层 KV 共享部分。

这里使用的简化规则是：

1. 早期层计算并缓存自己的 K/V 张量。
2. 后面层重用来自较早生产层的最新 K/V 张量。
3. 所有层仍然计算自己的查询投影。

这减少了随上下文长度增长的 KV 缓存数量。权衡是减少了模型容量，因为某些层不再拥有自己的 K/V 投影。

&nbsp;
## KV 共享内存节省

常规的 KV 缓存内存计算如下：

bytes = batch_size x seqlen x head_dim x n_kv_heads x n_layers x 2 (K,V) x bytes_per_elem

使用跨层 KV 共享时，我们将 `n_layers` 替换为 K/V 生产层的数量：

bytes = batch_size x seqlen x head_dim x n_kv_heads x n_kv_producing_layers x 2 (K,V) x bytes_per_elem

您可以使用此文件夹中的 [memory_estimator_kv_sharing.py](memory_estimator_kv_sharing.py) 脚本将其应用于不同的模型配置：

```bash
# Gemma 4 E2B-like setup
uv run memory_estimator_kv_sharing.py \
  --context_length 131072 \
  --emb_dim 2048 \
  --n_heads 8 \
  --n_layers 35 \
  --n_kv_groups 8 \
  --n_kv_producing_layers 15 \
  --batch_size 1 \
  --dtype bf16

# Gemma 4 E4B-like setup
# uv run memory_estimator_kv_sharing.py \
#   --context_length 131072 \
#   --emb_dim 2560 \
#   --n_heads 8 \
#   --n_layers 42 \
#   --n_kv_groups 4 \
#   --n_kv_producing_layers 24 \
#   --batch_size 1 \
#   --dtype bf16

==== Config ====
context_length         : 131072
emb_dim                : 2048
n_heads                : 8
n_layers               : 35
n_kv_groups            : 8
n_kv_producing_layers  : 15
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 256
GQA n_kv_heads         : 1

==== KV-cache totals across all layers ====
MHA total KV cache        : 37.58 GB
GQA total KV cache        : 4.70 GB
MHA + KV sharing          : 16.11 GB
GQA + KV sharing          : 2.01 GB
Ratio (MHA / GQA+sharing) : 18.67x
Savings vs MHA            : 94.64%
```

这是 Gemma 4 E2B 类型的设置。35 层包括 15 个 K/V 生产层，其余层重用较早的 K/V 张量。对于 E4B 类型设置，相应的数字是 42 个总层和 24 个 K/V 生产层。

下图显示了 E2B 类型设置的节省情况。为简单起见，这些图不包括来自滑动窗口注意力的额外节省。

&nbsp;

<img src="kv_memory_mha_gqa_kvsharing_gemma4_e2b.webp" alt="KV-sharing memory savings for Gemma 4 E2B-like setup" width="800px" />

&nbsp;

<img src="kv_memory_mha_gqa_kvsharing_gemma4_e4b.webp" alt="KV-sharing memory savings for Gemma 4 E4B-like setup" width="800px" />

&nbsp;

您可以通过以下方式重现类似的图：

```bash
uv run plot_memory_estimates_kv_sharing.py --preset gemma4_e2b
uv run plot_memory_estimates_kv_sharing.py --preset gemma4_e4b
```


&nbsp;
## KV 共享代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_sharing.py](gpt_with_kv_sharing.py) 脚本提供了比较常规 MHA 与跨层 KV 共享变体的实际示例。

查看实现细节的最简单方法是检查 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_sharing.py](gpt_with_kv_sharing.py) 之间的文件差异。注释故意保持相似，以便差异突出 KV 共更改。

请注意，该模型没有经过训练，因此生成无意义的文本。但是，您可以将其用作第 5-7 章中标准 GPT 模型的即用型替换，并进行训练。

此外，此实现使用了 [另一奖励部分](../03_kv-cache) 中解释的 KV 缓存，因此内存节省更加明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768
```

```bash
uv run gpt_with_kv_sharing.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--n_kv_producing_layers 6
```

在这个小型 GPT 设置中，整个模型仍然包含相同的全连接层和输出头。主要的内存差异在于多少注意力层在缓存中存储 K/V 张量。