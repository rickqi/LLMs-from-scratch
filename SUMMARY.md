# Summary

- [从零构建大型语言模型](README_CN.md)

---

# 学习笔记

- [学习路线与建议](docs/01-学习路线与建议.md)
- [环境配置](docs/02-环境配置.md)
- [LLM 生命周期与第 2 章预习](docs/03-LLM生命周期与第2章预习.md)
- [注意力机制](docs/04-注意力机制.md)
- [GPT 模型实现](docs/05-GPT模型实现.md)
- [预训练](docs/06-预训练.md)
- [分类微调](docs/07-分类微调.md)
- [指令微调](docs/08-指令微调.md)
- [PyTorch 入门](docs/09-PyTorch入门.md)
- [LoRA 参数高效微调](docs/10-LoRA参数高效微调.md)

---

# 环境配置

- [环境配置指南](setup/README_CN.md)
  - [Python 配置偏好](setup/01_optional-python-setup-preferences/README_CN.md)
  - [安装 Python 库](setup/02_installing-python-libraries/README_CN.md)
  - [Docker 环境](setup/03_optional-docker-environment/README_CN.md)
  - [AWS SageMaker 笔记本](setup/04_optional-aws-sagemaker-notebook/README_CN.md)

---

# 第 1 章：理解大型语言模型

- [第 1 章：理解大型语言模型](ch01/README_CN.md)

# 第 2 章：处理文本数据

- [第 2 章：处理文本数据](ch02/README_CN.md)
  - [主章节代码](ch02/01_main-chapter-code/README_CN.md)
  - [BPE 编码器比较](ch02/02_bonus_bytepair-encoder/README_CN.md)
  - [嵌入层 vs 线性层](ch02/03_bonus_embedding-vs-matmul/README_CN.md)
  - [数据加载器直觉理解](ch02/04_bonus_dataloader-intuition/README_CN.md)
  - [从零实现 BPE 分词器](ch02/05_bpe-from-scratch/README_CN.md)

# 第 3 章：编码注意力机制

- [第 3 章：编码注意力机制](ch03/README_CN.md)
  - [主章节代码](ch03/01_main-chapter-code/README_CN.md)
  - [高效多头注意力实现](ch03/02_bonus_efficient-multihead-attention/README_CN.md)
  - [理解 PyTorch Buffers](ch03/03_understanding-buffers/README_CN.md)

# 第 4 章：从零实现 GPT 模型

- [第 4 章：从零实现 GPT 模型](ch04/README_CN.md)
  - [主章节代码](ch04/01_main-chapter-code/README_CN.md)
  - [FLOPs 性能分析](ch04/02_performance-analysis/README_CN.md)
  - [KV 缓存](ch04/03_kv-cache/README_CN.md)
  - [分组查询注意力 (GQA)](ch04/04_gqa/README_CN.md)
  - [多头潜在注意力 (MLA)](ch04/05_mla/README_CN.md)
  - [滑动窗口注意力 (SWA)](ch04/06_swa/README_CN.md)
  - [混合专家模型 (MoE)](ch04/07_moe/README_CN.md)
  - [Gated DeltaNet](ch04/08_deltanet/README_CN.md)
  - [DeepSeek 稀疏注意力 (DSA)](ch04/09_dsa/README_CN.md)
  - [跨层 KV 共享](ch04/10_kv-sharing/README_CN.md)

# 第 5 章：在无标签数据上预训练

- [第 5 章：预训练](ch05/README_CN.md)
  - [主章节代码](ch05/01_main-chapter-code/README_CN.md)
  - [替代权重加载方法](ch05/02_alternative_weight_loading/README_CN.md)
  - [Gutenberg 数据集预训练](ch05/03_bonus_pretraining_on_gutenberg/README_CN.md)
  - [学习率调度器](ch05/04_learning_rate_schedulers/README_CN.md)
  - [超参数优化](ch05/05_bonus_hparam_tuning/README_CN.md)
  - [用户界面](ch05/06_user_interface/README_CN.md)
  - [GPT 转 Llama](ch05/07_gpt_to_llama/README_CN.md)
  - [内存高效权重加载](ch05/08_memory_efficient_weight_loading/README_CN.md)
  - [扩展 Tiktoken 分词器](ch05/09_extending-tokenizers/README_CN.md)
  - [PyTorch 性能优化](ch05/10_llm-training-speed/README_CN.md)
  - [Qwen3 从零实现](ch05/11_qwen3/README_CN.md)
  - [Gemma 3 从零实现](ch05/12_gemma3/README_CN.md)
  - [Olmo 3 从零实现](ch05/13_olmo3/README_CN.md)
  - [使用其他 LLM 替换](ch05/14_ch05_with_other_llms/README_CN.md)
  - [Tiny Aya 从零实现](ch05/15_tiny-aya/README_CN.md)
  - [Qwen3.5 从零实现](ch05/16_qwen3.5/README_CN.md)
  - [Gemma 4 从零实现](ch05/17_gemma4/README_CN.md)
  - [Muon 优化器](ch05/18_muon/README_CN.md)

# 第 6 章：文本分类微调

- [第 6 章：分类微调](ch06/README_CN.md)
  - [主章节代码](ch06/01_main-chapter-code/README_CN.md)
  - [额外实验](ch06/02_bonus_additional-experiments/README_CN.md)
  - [IMDb 分类](ch06/03_bonus_imdb-classification/README_CN.md)
  - [用户界面](ch06/04_user_interface/README_CN.md)

# 第 7 章：指令跟随微调

- [第 7 章：指令微调](ch07/README_CN.md)
  - [主章节代码](ch07/01_main-chapter-code/README_CN.md)
  - [数据集工具](ch07/02_dataset-utilities/README_CN.md)
  - [模型评估](ch07/03_model-evaluation/README_CN.md)
  - [DPO 偏好优化](ch07/04_preference-tuning-with-dpo/README_CN.md)
  - [数据集生成](ch07/05_dataset-generation/README_CN.md)
  - [用户界面](ch07/06_user_interface/README_CN.md)

---

# 附录

- [附录 A：PyTorch 入门](appendix-A/README_CN.md)
  - [主章节代码](appendix-A/01_main-chapter-code/README_CN.md)
  - [配置建议](appendix-A/02_setup-recommendations/README_CN.md)
- [附录 B：参考文献与扩展阅读](appendix-B/README_CN.md)
- [附录 C：练习解答](appendix-C/README_CN.md)
- [附录 D：训练循环增强](appendix-D/README_CN.md)
- [附录 E：LoRA 参数高效微调](appendix-E/README_CN.md)

---

# 视频字幕

- [02 - 环境配置](docs/youtube/02-环境配置.md)
- [03 - LLM 生命周期](docs/youtube/03-LLM生命周期.md)
- [04 - 文本数据处理](docs/youtube/04-文本数据处理.md)
- [05 - 注意力机制](docs/youtube/05-注意力机制.md)
- [06 - PyTorch Buffers](docs/youtube/06-PyTorch-Buffers.md)
- [07 - GPT 模型实现](docs/youtube/07-GPT模型实现.md)
- [08 - 预训练](docs/youtube/08-预训练.md)
- [09 - 分类微调](docs/youtube/09-分类微调.md)
- [10 - 指令微调](docs/youtube/10-指令微调.md)

---

# 其他

- [Python 包文档](pkg/llms_from_scratch/README_CN.md)
- [常见问题排查](troubleshooting_CN.md)
