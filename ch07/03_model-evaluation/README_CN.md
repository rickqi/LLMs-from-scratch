# 第 7 章：微调以遵循指令

此文件夹包含可用于模型评估的实用代码。

&nbsp;
## 使用 OpenAI API 评估指令响应

- [llm-instruction-eval-openai.ipynb](llm-instruction-eval-openai.ipynb) 笔记本使用 OpenAI 的 GPT-4 评估指令微调模型生成的响应。它处理以下格式的 JSON 文件：

```python
{
    "instruction": "What is the atomic number of helium?",
    "input": "",
    "output": "The atomic number of helium is 2.",               # <-- 测试集中给出的目标
    "model 1 response": "\nThe atomic number of helium is 2.0.", # <-- LLM 的响应
    "model 2 response": "\nThe atomic number of helium is 3."    # <-- 第二个 LLM 的响应
},
```

&nbsp;
## 使用 Ollama 本地评估指令响应

- [llm-instruction-eval-ollama.ipynb](llm-instruction-eval-ollama.ipynb) 笔记本提供上述方法的替代方案，利用通过 Ollama 下载的本地 Llama 3 模型。