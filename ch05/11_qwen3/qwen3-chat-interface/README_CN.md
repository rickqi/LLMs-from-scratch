# 从零开始构建 Qwen3 并带有聊天界面

这个奖励文件夹包含用于运行类似 ChatGPT 的用户界面以与预训练的 Qwen3 模型交互的代码。



![Chainlit UI 示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)



为了实现这个用户界面，我们使用开源的 [Chainlit Python 包](https://github.com/Chainlit/chainlit)。

&nbsp;
## 第 1 步：安装依赖项

首先，我们通过以下方式安装 `chainlit` 包和来自 [requirements-extra.txt](requirements-extra.txt) 列表的依赖项：

```bash
pip install -r requirements-extra.txt
```

或者，如果您使用 `uv`：

```bash
uv pip install -r requirements-extra.txt
```



&nbsp;

## 第 2 步：运行 `app` 代码

此文件夹包含 2 个文件：

1. [`qwen3-chat-interface.py`](qwen3-chat-interface.py)：此文件在思考模式下加载和使用 Qwen3 0.6B 模型。
2. [`qwen3-chat-interface-multiturn.py`](qwen3-chat-interface-multiturn.py)：与上述相同，但配置为记住消息历史。

（打开并检查这些文件以了解更多信息。）

从终端运行以下命令之一以启动 UI 服务器：

```bash
chainlit run qwen3-chat-interface.py
```

或者，如果您使用 `uv`：

```bash
uv run chainlit run qwen3-chat-interface.py
```

运行上述命令之一应该会打开一个新的浏览器标签，您可以在其中与模型交互。如果浏览器标签没有自动打开，请检查终端命令并将本地地址复制到您的浏览器地址栏中（通常，地址是 `http://localhost:8000`）。