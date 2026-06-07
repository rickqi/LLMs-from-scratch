# DeepSeek 稀疏注意力（DeepSeek Sparse Attention, DSA）

本奖励材料实现了 DeepSeek 稀疏注意力（DSA）机制，该机制在 [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) 中引入，并首先在实验性的 [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) 版本中发布。

下面的概述遵循 [从 DeepSeek V3 到 V3.2：架构、稀疏注意力和 RL 更新](https://magazine.sebastianraschka.com/p/technical-deepseek) 中对 DSA 的讨论。

&nbsp;
## 介绍

标准的因果自注意力为每个查询关注所有之前的标记，导致相对于序列长度 L 呈 O(L²) 计算和 O(L) KV 缓存增长。

[滑动窗口注意力（SWA）](../06_swa) 已经显示，将注意力限制在固定局部窗口会显著降低成本。在 SWA 中，每个查询标记只关注附近的前几个标记的局部跨度。

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/09.png" alt="Sliding window attention" width="800px" />

*图 1. 滑动窗口注意力将每个查询标记限制在固定的局部上下文窗口内。*

&nbsp;

DSA 使用相同的广泛思路，即只关注之前标记的子集。然而，它用学习的选择机制替换了固定窗口。对于每个查询标记，模型对候选过去标记进行评分，只保留最相关的那些。

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/10.png" alt="DeepSeek Sparse Attention selected-token pattern" width="800px" />

*图 2. DeepSeek 稀疏注意力为每个查询标记选择学习到的过去标记子集。*

&nbsp;

### 架构概述

DSA 在标准注意力之上添加了两个组件。

**1. Lightning Indexer（闪电索引器）**

对于每个查询标记 *t* 和每个候选过去标记 *s*，索引器计算标量相关性分数。此实现使参考代码中的比例因子变得明确：

$$I_{t,s} = \sum_{j=1}^{H_I} \frac{w_{t,j}}{\sqrt{H_I}} \cdot \text{ReLU}\left(\frac{q_{t,j} \cdot k_s}{\sqrt{d_I}}\right)$$

其中：
- $H_I$ 是轻量级索引头的数量，
- $q_{t,j}$ 是标记 *t* 和头 *j* 的索引器查询向量，
- $k_s$ 是过去标记 *s* 的共享索引器键向量，
- $w_{t,j}$ 是通过 $1 / \sqrt{H_I}$ 缩放的每头学习门控。

ReLU 将负点积贡献归零，门控总和跨索引头聚合，为每个过去标记生成单个相关性分数。

在完整的 DeepSeek 模型中，索引器来自多头潜注意力（MLA）的压缩标记表示。此文件夹保持 GPT 实现的简单性，并从常规隐藏状态计算索引器查询和键。

**2. Token Selector（标记选择器）**

计算完所有索引器分数后，只保留前 K 个最高分的位置。所有其他位置在标准 softmax 之前被掩码为 -∞，因此模型有效地只关注 $k \ll L$ 个标记。

索引器中的ReLU不是稀疏性的最终来源。由于分数在多个索引头上求和，大多数最终分数仍然可能非零。标记选择器通过只保留前 K 个位置来创建稀疏模式。

在生产融合实现中，这可以将注意力计算从 O(L²) 降低到 O(L·k)。此处的实现保持标准的密集注意力分数矩阵，并在 softmax 之前应用 DSA 选择的前 K 掩码。这使得选择逻辑易于检查，但它不提供融合核的计算节省。

下图总结了流程。闪电索引器对候选标记进行评分，选择器保留前 K 个位置，结果掩码限制常规注意力 softmax。

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/11.png" alt="DeepSeek Sparse Attention flowchart" width="700px" />

*图 3. DSA 首先对候选标记评分，然后为最终注意力掩码保留前 K 个标记。*

&nbsp;
## 实现

`gpt_with_kv_dsa.py` 提供：

| 类 | 描述 |
|---|---|
| `LightningIndexer` | 用于过去标记相关性的轻量级多头评分器。 |
| `MultiHeadAttentionWithDSA` | 带有 DSA 稀疏掩码 + 可选 KV 缓存的标准 MHA。 |
| `GPTModel` | 交换为 `MultiHeadAttentionWithDSA` 的 GPT 风格模型。 |

实现遵循此存储库中其他奖励材料的风格，可以作为独立脚本运行。它旨在让 DSA 机制在小型 GPT 风格模型中可检查。它不实现 DeepSeek 的完整 MLA 堆栈、融合稀疏核或部署特定优化。

&nbsp;
## 使用方法

```bash
uv run gpt_with_kv_dsa.py \
  --emb_dim 768 \
  --n_heads 12 \
  --n_layers 12 \
  --max_new_tokens 200 \
  --index_n_heads 4 \
  --index_head_dim 64 \
  --topk 64
```

关键参数：

| 参数 | 默认值 | 描述 |
|---|---|---|
| `--index_n_heads` | 4 | 轻量级索引头的数量（H_I）。 |
| `--index_head_dim` | 64 | 每个索引头维度。 |
| `--topk` | 64 | 每个查询关注的标记数（k）。对于短序列，限制在序列长度。 |

&nbsp;
## 与 DeepSeek V3.2 的关系

完整的 DeepSeek-V3.2 模型使用多头潜注意力（MLA，参见 [../05_mla](../05_mla)）和 DSA，索引器查询从共享压缩潜表示而不是原始输入派生。DeepSeek-V3.2 使用与 DeepSeek-V3.2-Exp 相同的架构，其中 DSA 首次引入和测试。

这里重现了关键选择思路。一个廉价的学习点积评分器在每个查询的注意力 softmax 之前将其限制为最相关的标记。

下面的推理成本比较报告提供了为什么 DSA 在长上下文部署中重要的有用背景。节省取决于生产核和服务基础设施，因此此图不应被视为此文件夹中教学实现的基准。

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/19.png" alt="Inference cost comparison for DeepSeek Sparse Attention" width="800px" />

*图 4. DeepSeek 在长上下文服务中报告的 DSA 推理成本节省，来自 [DeepSeek V3.2 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf)。*

&nbsp;
## 参考文献

- DeepSeek V3.2 技术报告：https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf
- DeepSeek V3.2-Exp 模型卡片和参考代码：https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- Sebastian Raschka 的 "从 DeepSeek V3 到 V3.2：架构、稀疏注意力和 RL 更新"：https://magazine.sebastianraschka.com/p/technical-deepseek