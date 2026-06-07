# 第 4 章：从零实现 GPT 模型以生成文本

## 主章节代码

- [01_main-chapter-code](01_main-chapter-code) 包含主章节代码

## 补充材料

- [02_performance-analysis](02_performance-analysis) 包含可选的代码，分析主章节中实现的 GPT 模型的性能
- [03_kv-cache](03_kv-cache) 实现了 KV 缓存（KV Cache），用于加速推理时的文本生成
- [07_moe](07_moe) 混合专家模型（Mixture-of-Experts, MoE）的说明与实现
- [ch05/07_gpt_to_llama](../ch05/07_gpt_to_llama) 包含将 GPT 架构实现转换为 Llama 3.2 的逐步指南，并加载 Meta AI 的预训练权重（在完成第 4 章后查看替代架构会很有意思，但也可以留到读完第 5 章后再看）

## 注意力机制的替代方案

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/attention-alternatives/attention-alternatives.webp">

- [04_gqa](04_gqa) 分组查询注意力（Grouped-Query Attention, GQA）入门 — 大多数现代 LLM（Llama 4、gpt-oss、Qwen3、Gemma 3 等）使用 GQA 替代常规多头注意力（MHA）
- [05_mla](05_mla) 多头潜在注意力（Multi-Head Latent Attention, MLA）入门 — DeepSeek V3 使用 MLA 替代常规 MHA
- [06_swa](06_swa) 滑动窗口注意力（Sliding Window Attention, SWA）入门 — Gemma 3 等模型使用
- [08_deltanet](08_deltanet) 门控 DeltaNet（Gated DeltaNet）说明 — 一种流行的线性注意力变体（Qwen3-Next 和 Kimi Linear 使用）
- [10_kv-sharing](10_kv-sharing) 跨层 KV 共享（Cross-Layer KV Sharing）入门 — Gemma 4 E2B 和 E4B 使用此技术减少 KV 缓存内存

## 更多资源

在下面的视频中，我提供了一段跟练（code-along）视频，作为本章内容的补充学习材料：

[![视频链接](https://img.youtube.com/vi/YSAkgEarBGE/0.jpg)](https://www.youtube.com/watch?v=YSAkgEarBGE)
