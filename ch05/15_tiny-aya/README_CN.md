# 从零开始实现 Tiny Aya 3.35B

Tiny Aya 是 Cohere 推出的一个新"小"LLM，据说在 3B 参数规模上是"最有能力的多语言开放权重模型"。（根据 [公告帖](https://cohere.com/blog/cohere-labs-tiny-aya)，Tiny Aya 优于 Qwen3-4B、Gemma 3 4B 和 Ministral 3 3B）。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/tiny-aya/01.webp">

这是一个在本地运行和实验的好模型。唯一的警告是，虽然它是一个开放权重模型，但其许可条款相对严格，仅允许非商业使用。

除此之外，Arya 是一个 3.35B 参数模型，有多种变体可用于个人和（非商业）研究：

  - [tiny-aya-base](https://huggingface.co/CohereLabs/tiny-aya-base)（基础模型）
  - [tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global)（在语言和区域之间取得最佳平衡；笔记本默认）
  - [tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire)（针对南亚语言优化）
  - [tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water)（针对欧洲和亚太语言优化）
  - [tiny-aya-earth](https://huggingface.co/CohereLabs/tiny-aya-earth)（针对西亚和非洲语言优化）


更具体地说，以下是模型所针对的语言列表：

| Region           | Languages                                                    | Optimized Model |
| ---------------- | ------------------------------------------------------------ | --------------- |
| **Asia Pacific** | Traditional Chinese, Cantonese, Vietnamese, Tagalog, Javanese, Khmer, Thai, Burmese, Malay, Korean, Lao, Indonesian, Simplified Chinese, Japanese | tiny-aya-water  |
| **Africa**       | Zulu, Amharic, Hausa, Igbo, Swahili, Xhosa, Wolof, Shona, Yoruba, Nigerian Pidgin, Malagasy | tiny-aya-earth  |
| **South Asia**   | Telugu, Marathi, Bengali, Tamil, Hindi, Punjabi, Gujarati, Urdu, Nepali | tiny-aya-fire   |
| **Europe**       | Catalan, Galician, Dutch, Danish, Finnish, Czech, Portuguese, French, Lithuanian, Slovak, Basque, English, Swedish, Polish, Spanish, Slovenian, Ukrainian, Greek, Bokmål, Romanian, Serbian, German, Italian, Russian, Irish, Hungarian, Bulgarian, Croatian, Estonian, Latvian, Welsh | tiny-aya-water  |
| **West Asia**    | Arabic, Maltese, Turkish, Hebrew, Persian                    | tiny-aya-earth  |

从架构上看，Tiny Aya 是一个经典的解码器式 transformer，有一些值得注意的修改（除了明显的如 SwiGLU 和分组查询注意力）：

1. **并行 transformer 块。** 并行 transformer 块从相同的规范化输入计算注意力和 MLP，然后将两者一次性添加到残差中。我假设这是为了减少层内的串行依赖，以计算吞吐量。

2. **滑动窗口注意力。** 具体来说，它使用 3:1 的局部：全局比例，类似于 Arcee Trinity 和 Olmo 3。窗口大小也是 4096。此外，类似于 Arcee，滑动窗口层使用 RoPE，而全注意力层使用 NoPE。

3. **LayerNorm。** 大多数架构已转向 RMSNorm，因为它计算成本稍低且表现良好。Tiny Aya 更经典，使用修改后的 LayerNorm 版本（这里的实现类似于标准 LayerNorm，但没有偏置参数）。

&nbsp;
## 文件

[standalone-tiny-aya.ipynb](standalone-tiny-aya.ipynb) 是一个独立的 Jupyter 笔记本，实现了 Tiny Aya 架构并加载预训练权重。


替代的 [standalone-tiny-aya-plus-kvcache.ipynb](standalone-tiny-aya-plus-kv-cache.ipynb) 笔记本添加了 KV 缓存以获得更好的运行时性能（但增加了更多的代码复杂性）。要了解更多关于 KV 缓存的信息，请参阅我的 [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) 文章。


<br>

要了解更多关于架构差异并了解与其他架构的比较，请参阅我的 [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) 文章。