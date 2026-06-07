# 第 5 章：在无标签数据上预训练

## 主章节代码

- [01_main-chapter-code](01_main-chapter-code) 包含主章节代码

## 补充材料

- [02_alternative_weight_loading](02_alternative_weight_loading) 包含从其他来源加载 GPT 模型权重的代码，以防 OpenAI 的模型权重不可用
- [03_bonus_pretraining_on_gutenberg](03_bonus_pretraining_on_gutenberg) 包含在 Project Gutenberg 全部书籍语料库上更长时间预训练 LLM 的代码
- [04_learning_rate_schedulers](04_learning_rate_schedulers) 包含实现更复杂训练函数的代码，包括学习率调度器（Learning Rate Schedulers）和梯度裁剪（Gradient Clipping）
- [05_bonus_hparam_tuning](05_bonus_hparam_tuning) 包含可选的超参数调优脚本
- [06_user_interface](06_user_interface) 实现了一个交互式用户界面，用于与预训练后的 LLM 进行对话
- [08_memory_efficient_weight_loading](08_memory_efficient_weight_loading) 包含一个补充 Notebook，展示如何通过 PyTorch 的 `load_state_dict` 方法更高效地加载模型权重
- [09_extending-tokenizers](09_extending-tokenizers) 包含 GPT-2 BPE 分词器的从零实现
- [10_llm-training-speed](10_llm-training-speed) 展示提升 LLM 训练速度的 PyTorch 性能优化技巧
- [18_muon](18_muon) 说明如何在 GPT 模型训练中使用 Muon 优化器

## 从零实现各主流 LLM 架构

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">

- [07_gpt_to_llama](07_gpt_to_llama) 将 GPT 架构转换为 Llama 3.2 的逐步指南，并加载 Meta AI 的预训练权重
- [11_qwen3](11_qwen3) Qwen3 0.6B 和 Qwen3 30B-A3B（混合专家模型）的从零实现，包括加载基础版、推理版和编程版模型预训练权重的代码
- [12_gemma3](12_gemma3) Gemma 3 270M 及其 KV 缓存版本的从零实现，包括加载预训练权重的代码
- [13_olmo3](13_olmo3) Olmo 3 7B 和 32B（Base、Instruct 和 Think 变体）及其 KV 缓存版本的从零实现，包括加载预训练权重的代码
- [17_gemma4](17_gemma4) Gemma 4 的 E2B 和 E4B 稠密变体的从零实现

## 本章跟练视频

[![视频链接](https://img.youtube.com/vi/Zar2TJv-sE0/0.jpg)](https://www.youtube.com/watch?v=Zar2TJv-sE0)
