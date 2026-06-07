# 从零开始实现 Gemma 3 270M

此文件夹中的此 [standalone-gemma3.ipynb](standalone-gemma3.ipynb) Jupyter 笔记本包含 Gemma 3 270M 的从头实现。运行需要大约 2 GB RAM。

替代的 [standalone-gemma3-plus-kvcache.ipynb](standalone-gemma3-plus-kvcache.ipynb) 笔记本添加了 KV 缓存以获得更好的运行时性能（但增加了更多的代码复杂性）。要了解更多关于 KV 缓存的信息，请参阅我的 [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) 文章。

| Model             | Mode              | Hardware        | Tokens/sec | GPU Memory (VRAM) |
| ----------------- | ----------------- | --------------- | ---------- | ----------------- |
| Gemma3Model 270M  | Regular           | Mac Mini M4 CPU | 8          | -                 |
| Gemma3Model 270M  | Regular compiled  | Mac Mini M4 CPU | 9          | -                 |
| Gemma3Model 270M  | KV cache          | Mac Mini M4 CPU | 130        | -                 |
| Gemma3Model 270M  | KV cache compiled | Mac Mini M4 CPU | 224        | -                 |
|                   |                   |                 |            |                   |
| Gemma3Model 270M  | Regular           | Mac Mini M4 GPU | 16         | -                 |
| Gemma3Model 270M  | Regular compiled  | Mac Mini M4 GPU | Error      | -                 |
| Gemma3Model 270M  | KV cache          | Mac Mini M4 GPU | 23         | -                 |
| Gemma3Model 270M  | KV cache compiled | Mac Mini M4 GPU | Error      | -                 |
|                   |                   |                 |            |                   |
| Gemma3Model 270M  | Regular           | Nvidia A100 GPU | 28         | 1.84 GB           |
| Gemma3Model 270M  | Regular compiled  | Nvidia A100 GPU | 128        | 2.12 GB           |
| Gemma3Model 270M  | KV cache          | Nvidia A100 GPU | 26         | 1.77 GB           |
| Gemma3Model 270M  | KV cache compiled | Nvidia A100 GPU | 99         | 2.12 GB           |


下面是与 Qwen3 0.6B 并排比较的参考模型；如果您有兴趣了解 Qwen3 0.6B 独立笔记本，可以在 [此处](../11_qwen3) 找到。

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gemma3/gemma3-vs-qwen3.webp">

<br>

要了解更多关于架构差异并了解与其他架构的比较，请参阅我的 [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) 文章。