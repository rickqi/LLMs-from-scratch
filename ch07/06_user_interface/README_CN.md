# 构建指令微调 GPT 模型的用户界面


此奖励文件夹包含运行 ChatGPT 风格用户界面的代码，用于与第 7 章的指令微调 GPT 进行交互，如下图所示。

![Chainlit UI 示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-sft.webp?2)

要实现此用户界面，我们使用开源的 [Chainlit Python 包](https://github.com/Chainlit/chainlit)。

&nbsp;
## 步骤 1：安装依赖项

首先，我们通过以下方式安装 `chainlit` 包：

```bash
pip install chainlit
```

（或者，执行 `pip install -r requirements-extra.txt`。）

&nbsp;
## 步骤 2：运行 `app` 代码

[`app.py`](app.py) 文件包含基于的 UI 代码。打开并检查这些文件以了解更多信息。

此文件加载并使用我们在第 7 章生成的 GPT-2 权重。这要求您首先执行 [`../01_main-chapter-code/ch07.ipynb`](../01_main-chapter-code/ch07.ipynb) 文件。

从终端执行以下命令以启动 UI 服务器：

```bash
chainlit run app.py
```

运行上述命令应该会打开一个新的浏览器标签页，您可以在其中与模型交互。如果浏览器标签页没有自动打开，请检查终端命令并将本地地址复制到浏览器地址栏（通常，地址是 `http://localhost:8000`）。