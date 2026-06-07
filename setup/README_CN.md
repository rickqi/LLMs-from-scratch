# 可选环境配置指南

本文档列出了配置你的计算机以使用本仓库代码的不同方法。建议从上到下浏览各个部分，然后选择最适合你需求的方法。

## 快速开始

如果你已经在机器上安装了 Python，最快的开始方式是从本代码仓库的根目录执行以下 pip 安装命令，安装 [../requirements.txt](../requirements.txt) 文件中列出的依赖包：

```bash
pip install -r requirements.txt
```

> **注意：** 如果你在 Google Colab 上运行任何 Notebook 并需要安装依赖，只需在 Notebook 顶部的新的代码单元格中运行：
> `pip install uv && uv pip install --system -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`
> 另外，在克隆仓库后，你可以通过从项目根目录执行 `uv pip install --group bonus` 来安装所有补充材料的依赖。当你在后续查看可选的补充材料时，这样就无需单独安装了。

在下面的视频中，我分享了我个人在电脑上配置 Python 环境的方法：

[![视频链接](https://img.youtube.com/vi/yAcWnfsZhzo/0.jpg)](https://www.youtube.com/watch?v=yAcWnfsZhzo)

## 本地环境配置

本节提供了在本地运行本书代码的建议。请注意，本书主章节的代码设计为可在普通笔记本电脑上在合理时间内运行，不需要专门的硬件。我在一台 M3 MacBook Air 笔记本电脑上测试了所有主章节。此外，如果你的笔记本或台式机有 NVIDIA GPU，代码会自动利用它。

### 配置 Python

如果你还没有在机器上配置 Python，我在以下目录中写了我个人的 Python 配置偏好：

- [01_optional-python-setup-preferences](./01_optional-python-setup-preferences)
- [02_installing-python-libraries](./02_installing-python-libraries)

下面的 *使用 DevContainers* 部分概述了一种在机器上安装项目依赖的替代方法。

### 使用 Docker DevContainers

作为上述 *配置 Python* 部分的替代方案，如果你偏好将项目依赖和配置隔离的开发环境，使用 Docker 是一种高效的解决方案。这种方法无需手动安装软件包和库，并确保开发环境一致。你可以在以下位置找到更多关于配置 Docker 和使用 DevContainer 的说明：

- [03_optional-docker-environment](03_optional-docker-environment)

### Visual Studio Code 编辑器

有很多优秀的代码编辑器可选。我个人偏好广泛使用的开源编辑器 [Visual Studio Code (VSCode)](https://code.visualstudio.com)，它可以通过许多有用的插件和扩展轻松增强（详见下方的 *VSCode 扩展* 部分）。macOS、Linux 和 Windows 的下载说明可在 [VSCode 官网](https://code.visualstudio.com) 找到。

### VSCode 推荐扩展

如果你使用 Visual Studio Code (VSCode) 作为主要代码编辑器，可以在 `.vscode` 子文件夹中找到推荐扩展。这些扩展提供了对本仓库有帮助的增强功能和工具。

安装方法：在 VSCode 中打开此 "setup" 文件夹（File → Open Folder...），然后点击右下角弹出菜单中的 "Install" 按钮。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/vs-code-extensions.webp?1" alt="1" width="700">

或者，你可以将 `.vscode` 扩展文件夹移动到本 GitHub 仓库的根目录：

```bash
mv setup/.vscode ./
```

这样，每次打开 `LLMs-from-scratch` 主文件夹时，VSCode 会自动检查推荐扩展是否已在你的系统上安装。

## 云资源

本节介绍运行本书代码的云替代方案。

虽然代码可以在不带独立 GPU 的普通笔记本和台式机上运行，但带有 NVIDIA GPU 的云平台可以显著缩短代码运行时间，尤其是在第 5 到第 7 章中。

### 使用 Lightning Studio

为了获得流畅的云端开发体验，我推荐 [Lightning AI Studio](https://lightning.ai/) 平台。它允许用户建立持久化环境，并在云端 CPU 和 GPU 上同时使用 VSCode 和 Jupyter Lab。

启动新的 Studio 后，打开终端并执行以下设置步骤来克隆仓库并安装依赖：

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
pip install -r requirements.txt
```

（与 Google Colab 不同：这些步骤只需执行一次，因为 Lightning AI Studio 的环境是持久化的，即使你在 CPU 和 GPU 机器之间切换也是如此。）

然后，导航到你想运行的 Python 脚本或 Jupyter Notebook。另外，你还可以轻松连接 GPU 来加速代码运行——例如在第 5 章预训练 LLM 或第 6、7 章微调时。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/studio.webp" alt="1" width="700">

### 使用 Google Colab

要使用 Google Colab 云环境，请前往 [https://colab.research.google.com/](https://colab.research.google.com/)，从 GitHub 菜单打开相应章节的 Notebook，或者将 Notebook 拖入 *Upload* 字段，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_1.webp" alt="1" width="700">

同时，请确保将 Notebook 需要导入的相关文件（数据集文件和 .py 文件）也上传到 Colab 环境中，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_2.webp" alt="2" width="700">

你可以通过更改 *Runtime* 来选择在 GPU 上运行代码，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_3.webp" alt="3" width="700">

## 有问题？

如果你有任何问题，请随时通过本 GitHub 仓库的 [Discussions](https://github.com/rasbt/LLMs-from-scratch/discussions) 论坛联系我们。
