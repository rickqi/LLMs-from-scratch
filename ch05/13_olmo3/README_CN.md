# 从零开始实现 Olmo 3 7B 和 32B

此文件夹中的此 [standalone-olmo3.ipynb](standalone-olmo3.ipynb) Jupyter 笔记本包含 Olmo 3 7B 和 32B 的从头实现，运行需要大约 13 GB RAM。

替代的 [standalone-olmo3-plus-kvcache.ipynb](standalone-olmo3-plus-kv-cache.ipynb) 笔记本添加了 KV 缓存以获得更好的运行时性能（但增加了更多的代码复杂性）。要了解更多关于 KV 缓存的信息，请参阅我的 [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) 文章。

下面是与 Qwen3 并排比较的参考模型；如果您有兴趣了解 Qwen3 0.6B 独立笔记本，可以在 [此处](../11_qwen3) 找到。

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-7B.webp?1">

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-32B.webp?1">

Olmo 3 还有不同的变体，如下所示（架构相同，仅训练流程不同）：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-pipeline.webp?1">


&nbsp;
## Olmo 3 与 Qwen3 的比较

专注于架构而非训练细节，本节提供与 Qwen3 的简要比较。


7B 模型：

1. 正如我们在上面的图中看到的，Olmo 3 架构与 Qwen3 相对相似。然而，值得注意的是，这基本上很可能是受 Olmo 2 前身启发，而不是 Qwen3。

2) 与 Olmo 2 类似，Olmo 3 仍使用后规范（post-norm）而非前规范（pre-norm），正如他们在 Olmo 2 论文中发现的那样，它稳定了训练。

3) 有趣的是，7B 模型仍然使用类似 Olmo 2 的多头注意力。但是，为了提高效率并减少 KV 缓存大小，他们现在使用滑动窗口注意力（例如，类似 Gemma 3）。

接下来是 32B 模型：

4) 总体而言，它是相同的架构，只是规模扩大了。此外，比例（例如，从输入到前馈层中间的大小等）大致与 Qwen3 中的比例相匹配。

5) 我的猜测是，架构最初由于词汇量较小而比 Qwen3 小一些，然后他们将中间大小扩展从 Qwen3 的 5 倍扩大到 Olmo 3 的 5.4 倍，以获得 32B 模型进行直接比较。

6) 另外，请注意 32B 模型（终于！）使用了分组查询注意力。



<br>

要了解更多关于架构差异并了解与其他架构的比较，请参阅我的 [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) 文章。