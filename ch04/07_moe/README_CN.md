# 混合专家（Mixture of Experts, MoE）

本奖励材料说明了在使用混合专家（MoE）层而不是常规前馈（FFN）层时的内存节省（每标记）。

&nbsp;
## 介绍

MoE 中的核心思想是用多个专家层替换 transformer 块中的每个前馈模块，其中这些专家层中的每一个也是一个前馈模块。这意味着我们将单个前馈块替换为多个前馈块，如下图所示。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/1.webp" alt="SWA" width="800px" />

transformer 块中的前馈块（上图中的深灰色块）通常包含模型总参数的大量。（注意 transformer 块，以及前馈块，在 LLM 中重复很多次；在 DeepSeek-V3 的情况下，重复 61 次。）

因此，在 MoE 设置中，将*单个*前馈块替换为*多个*前馈块会显著增加模型的总参数计数。但是，关键的技巧是，我们不（"激活"）所有专家来处理每个标记。相反，路由器（router）只为每个标记选择一小部分专家。

因为只有几个专家同时处于活动状态，MoE 模块通常被称为*稀疏*（sparse），而总是使用完整参数集的模块被称为*密集*（dense）模块。然而，通过 MoE 增加的大量总参数提高了 LLM 的能力，这意味着它在训练过程中可以吸收更多知识。然而，推理时的稀疏性保持高效，因为我们不会同时使用所有参数。

例如，DeepSeek-V3 每个 MoE 模块有 256 个专家，总共有 6710 亿参数。然而，在推理过程中，每次只有 9 个专家处于活动状态（1 个共享专家加上路由器选择的 8 个）。这意味着每个标记推理步骤只使用 370 亿参数，而不是所有 6710 亿。

DeepSeek-V3 MoE 设计的一个显著特征是使用共享专家。这是一个为每个标记始终活动的专家。这个想法并不新颖，早在 2022 年的 [DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) 和 2024 年的 [DeepSeek MoE](https://arxiv.org/abs/2401.06066) 论文中就引入了。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/3.webp?1" alt="MoE shared expert" width="500px" />

（来自 [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) 论文的注释图。）

&nbsp;

具有共享专家的好处首先在 [DeepSpeed-MoE 论文](https://arxiv.org/abs/2201.05596) 中注意到，他们发现与没有共享专家相比，它提高了整体建模性能。这可能是因为常见的或重复的模式不需要由多个单独的专家学习，这为他们留下了更多空间来学习更专业的模式。

&nbsp;
## 混合专家（MoE）内存节省

MoE 模型中的内存节省主要来自减少的激活存储和计算。在常规（密集）前馈层（FFN）中，每个标记激活完整的中间维度。

相比之下，MoE 层将每个标记路由到仅一小部分专家（例如，每个标记 `top_k` 出 `num_experts`）。

使用 MoE 层时，每个标记只有 `top_k` 专家处于活动状态，因此有效的内存（和计算）相对于相同总容量的密集 FFN 大致按 `top_k / num_experts` 的比例缩放。

您可以使用此文件夹中的 [memory_estimator_moe.py](memory_estimator_moe.py) 脚本为不同的模型配置应用此公式，以查看通过使用 MoE 而不是 FFN 可以节省多少内存（注意，这是针对单个 transformer 块，要获得总节省，乘以模型中的 transformer 块数量）：

```bash
uv run memory_estimator_moe.py --emb_dim 7168 --hidden_dim 14336 --ffn_type swiglu \
  --num_experts 8 --top_k 2 --match_dense 
==== Config ====
emb_dim                : 7168
hidden_size            : 14336
ffn_type               : swiglu
num_experts            : 8
top_k                  : 2
dtype                  : bf16 (2 Bytes/elem)
match_dense            : True

==== Model weights (parameters) ====
Dense FFN params       : 308,281,344 (0.62 GB)
Per-expert params      : 38,535,168 (0.08 GB)
Router params          : 57,344 (0.00 GB)
MoE TOTAL params       : 308,338,688 (0.62 GB)
MoE ACTIVE/Token       : 77,127,680 (0.15 GB)
moe_hidden_size        : 1792
```

因此，根据上面的结果，我们可以看到，如果我们有一个输入/输出维度（`emb_dim`）为 7,168，中间大小（`hidden_dim`）为 14,336 的 FFN，这个层有约 308M 参数，所有这些参数在前向传播中都处于活动状态。

现在，如果我们使用具有大致相同总数参数（~308M）的 MoE 层，有 8 个专家，其中 2 个专家处于活动状态，每个前向传播中只有约 77M 参数处于活动状态。

而且，在专家数量恒定的情况下，我们拥有的专家越多，活动参数的数量就越少，"节省"就越大：

&nbsp;


&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/2.webp" alt="SWA" width="500px" />

&nbsp;

您可以通过以下方式重现该图：

```bash
uv run plot_memory_estimates_moe.py \
    --emb_dim 7168 \
    --hidden_dim 28672 \
    --ffn_type swiglu \
    --top_k 8
```


&nbsp;
## MoE 代码示例

此文件夹中的 [gpt_with_kv_ffn.py](gpt_with_kv_ffn.py) 和 [gpt_with_kv_moe.py](gpt_with_kv_moe.py) 脚本提供了在 GPT 模型实现的上下文中比较常规 FFN 和 MoE 内存使用的实际示例。请注意，两个脚本都使用 [SwiGLU](https://arxiv.org/abs/2002.05202) 前馈模块，如本页第一图所示（GPT-2 传统上使用 GELU）。

**注意：该模型没有经过训练，因此生成无意义的文本。您可以在奖励材料中找到训练过的 MoE，[../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb](../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb)。**


首先，让我们用常规 FFN 运行模型：

```bash
uv run gpt_with_kv_ffn.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 32768

...
Avg FFN time/call: 0.759 ms
Avg FFN mem delta/call: 0.19 MB (max 0.75 MB)
...
Time: 25.13 sec
40 tokens/sec
Max memory allocated: 11.47 GB
```

为了与 MoE 进行公平比较，我们必须缩小专家大小。例如，如果我们使用 32 个专家，我们必须设置 `--hidden_dim 32768/32`：

```bash
uv run gpt_with_kv_moe.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 1024 \
--num_experts 32 \
--num_experts_per_tok 2

...
Avg MoE FF time/call: 1.555 ms
Avg MoE FF mem delta/call: 0.04 MB (max 0.11 MB)
...
Time: 35.11 sec
29 tokens/sec
Max memory allocated: 11.48 GB
```

我们可以看到，密集前馈层处理一个标记大约需要 0.76 毫秒，并使用约 0.19 MB 的激活（峰值接近 0.75 MB），

稀疏 MoE 层只保留约 0.04 MB 的内存（峰值 0.11）。然而，这大致需要两倍的计算时间。（还有额外的路由开销，我的实现也可能不是最有效的。）

总体生成仍然在两种情况下都达到约 11.5 GB 的 GPU 内存峰值，因为两个版本都加载相同数量的权重参数，并且具有相同的 KV 缓存大小，这些在这里占主导地位。

无论如何，我们可以看到这里的权衡，MoE 将 FFN 内存减少了约 4-5 倍，同时大致将前馈计算时间加倍。

注意，如果我们一次处理更多标记，例如批量大小大于 1（这里由于代码简单性我们没有批次），节省会更加明显。