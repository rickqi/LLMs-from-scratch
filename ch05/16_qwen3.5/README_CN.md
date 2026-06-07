# 从零开始实现 Qwen3.5 0.8B

此文件夹包含 [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) 的从头风格实现。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen3.5/03.webp">

Qwen3.5 基于 Qwen3-Next 架构，我在我的 [Beyond Standard LLMs](https://magazine.sebastianraschka.com/p/beyond-standard-llms) 文章的 [2. (线性) 注意力混合](https://magazine.sebastianraschka.com/i/177848019/2-linear-attention-hybrids) 部分中更详细地描述了它。

<a href="https://magazine.sebastianraschka.com/p/beyond-standard-llms"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen3.5/02.webp" width="500px"></a>

请注意，Qwen3.5 交替使用 `linear_attention` 和 `full_attention` 层。  
笔记本在重用来自 [qwen3_5_transformers.py](qwen3_5_transformers.py) 的线性注意力构建块的同时，保持了完整模型流程的可读性，其中包含在 Apache 2.0 开源许可下的 Hugging Face 线性注意力代码。

&nbsp;
## 文件

- [qwen3.5.ipynb](qwen3.5.ipynb): 主要的 Qwen3.5 0.8B 笔记本实现。
- [qwen3.5-plus-kv-cache.ipynb](qwen3.5-plus-kv-cache.ipynb): 带有 KV 缓存解码的相同模型，以提高效率。
- [qwen3_5_transformers.py](qwen3_5_transformers.py): 用于 Qwen3.5 线性注意力的 Hugging Face Transformers 的一些辅助组件。