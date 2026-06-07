# 第 5 章：在无标签数据上预训练

### 主章节代码

- [ch05.ipynb](ch05.ipynb) 包含本章中出现的所有代码
- [previous_chapters.py](previous_chapters.py) 是一个 Python 模块，包含前几章的 `MultiHeadAttention` 模块和 `GPTModel` 类，我们在 [ch05.ipynb](ch05.ipynb) 中导入它来预训练 GPT 模型
- [gpt_download.py](gpt_download.py) 包含用于下载预训练 GPT 模型权重的工具函数
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章的练习解答

### 可选代码

- [gpt_train.py](gpt_train.py) 是一个独立的 Python 脚本文件，包含我们在 [ch05.ipynb](ch05.ipynb) 中实现的 GPT 模型训练代码（可将其视为本章的代码总结）
- [gpt_generate.py](gpt_generate.py) 是一个独立的 Python 脚本文件，包含我们在 [ch05.ipynb](ch05.ipynb) 中实现的加载和使用 OpenAI 预训练模型权重的代码
