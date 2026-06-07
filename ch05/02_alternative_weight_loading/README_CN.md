# 预训练权重的替代加载方法

此文件夹包含在 OpenAI 权限不可用时的替代权重加载策略。

- [weight-loading-pytorch.ipynb](weight-loading-pytorch.ipynb):（推荐）包含代码，用于从 PyTorch 状态字典加载权重，这些状态字典是我通过转换原始 TensorFlow 权限创建的

- [weight-loading-hf-transformers.ipynb](weight-loading-hf-transformers.ipynb): 包含代码，用于通过 `transformers` 库从 Hugging Face Model Hub 加载权重

- [weight-loading-hf-safetensors.ipynb](weight-loading-hf-safetensors.ipynb): 包含代码，用于通过 `safetensors` 库直接从 Hugging Face Model Hub 加载权重（跳过 Hugging Face transformer 模型的实例化）