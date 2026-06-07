# 安装本书中使用的 Python 包和库

本文档提供了有关检查已安装 Python 版本和包的更多信息。（有关安装 Python 和 Python 包的更多信息，请参阅 [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) 文件夹。）

我在本书中使用了在 [这里](https://github.com/rasbt/LLMs-from-scratch/blob/main/requirements.txt) 列出的以下库。这些库的更新版本也可能兼容。但是，如果您在使用代码时遇到任何问题，可以尝试这些库版本作为备选方案。


> **注意：**
> 如果您按照 [选项 1：使用 uv](../01_optional-python-setup-preferences/README.md) 中描述使用 `uv`，您可以在下面的命令中将 `pip` 替换为 `uv pip`。例如，`pip install -r requirements.txt` 变为 `uv pip install -r requirements.txt`


要最方便地安装这些要求，您可以使用此代码存储库根目录中的 `requirements.txt` 文件并执行以下命令：

```bash
pip install -r requirements.txt
```

或者，您可以通过 GitHub URL 如下安装：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/requirements.txt
```


然后，完成安装后，请使用以下命令检查是否所有包都已安装且是最新的版本：

```bash
python python_environment_check.py
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_1.jpg" width="600px">

还建议通过在此目录中运行 `python_environment_check.ipynb` 在 JupyterLab 中检查版本，这应该会给出与上述相同的结果。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_2.jpg" width="500px">

如果您看到以下问题，您的 JupyterLab 实例可能连接到了错误的 conda 环境：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/jupyter-issues.jpg" width="450px">

在这种情况下，您可能希望使用 `watermark` 并使用 `--conda` 标志检查您是否在正确的 conda 环境中打开了 JupyterLab 实例：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/watermark.jpg" width="350px">


&nbsp;
## 安装 PyTorch

PyTorch 可以像安装任何其他 Python 库或包一样使用 pip 安装。例如：

```bash
pip install torch
```

但是，由于 PyTorch 是一个包含 CPU 和 GPU 兼容代码的全面库，安装可能需要额外的设置和说明（更多信息请参见书中的 *A.1.3 安装 PyTorch*）。

还强烈建议在官方 PyTorch 网站上的 [https://pytorch.org](https://pytorch.org) 查看安装指南菜单。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/pytorch-installer.jpg" width="600px">

<br>



&nbsp;
## JupyterLab 技巧

如果您在 JupyterLab 而不是 VSCode 中查看笔记本代码，请注意 JupyterLab（在其默认设置中）在最近的版本中存在滚动错误。我的建议是转到设置 -> 设置编辑器并将"窗口模式"更改为"无"（如下图所示），这似乎可以解决问题。


![Jupyter 问题 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter 问题 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)

<br>

---

有任何问题？请在 [讨论论坛](https://github.com/rasbt/LLMs-from-scratch/discussions) 中随时与我们联系。