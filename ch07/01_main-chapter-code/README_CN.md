# 第 7 章：指令跟随微调

### 主章节代码

- [ch07.ipynb](ch07.ipynb) 包含本章中出现的所有代码
- [previous_chapters.py](previous_chapters.py) 是一个 Python 模块，包含我们在前几章中编写和训练的 GPT 模型以及许多工具函数，我们在本章中复用它们
- [gpt_download.py](gpt_download.py) 包含用于下载预训练 GPT 模型权重的工具函数
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章的练习解答

### 可选代码

- [load-finetuned-model.ipynb](load-finetuned-model.ipynb) 是独立的 Jupyter Notebook，用于加载我们在本章中创建的指令微调模型
- [gpt_instruction_finetuning.py](gpt_instruction_finetuning.py) 是独立的 Python 脚本，用于按照主章节描述进行指令微调（可将其视为专注于微调部分的本章总结）

用法：

```bash
python gpt_instruction_finetuning.py
```

- [ollama_evaluate.py](ollama_evaluate.py) 是独立的 Python 脚本，用于按主章节描述评估微调模型的响应（可将其视为专注于评估部分的本章总结）

用法：

```bash
python ollama_evaluate.py --file_path instruction-data-with-response-standalone.json
```

- [exercise_experiments.py](exercise_experiments.py) 是可选脚本，实现练习解答；详见 [exercise-solutions.ipynb](exercise-solutions.ipynb)
