# Python 设置技巧



安装 Python 和设置计算环境有几种方法。在这里，我分享我的个人偏好。

<br>

> **注意：**
> 如果您在 Google Colab 上运行任何笔记本并想要安装依赖项，只需在笔记本顶部的单元格中运行以下代码并跳过本教程的其余部分：
> `pip install uv && uv pip install --system -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`

下面的其余部分描述了如何在您的本地机器上管理 Python 环境和包。

我一直是 [Conda](https://anaconda.org/anaconda/conda) 和 [pip](https://pypi.org/project/pip/) 的长期用户，但最近，[uv](https://github.com/astral-sh/uv) 包获得了显著的普及，因为它提供了更快、更高效的包安装和依赖项解析方法。

我建议从 *选项 1：使用 uv* 开始，因为这是 2025 年更现代的方法。如果您在 *选项 1* 中遇到问题，请考虑 *选项 2：使用 Conda*。

在本教程中，我使用的是运行 macOS 的计算机，但此工作流程与 Linux 机器相似，也可能适用于其他操作系统。

&nbsp;
# 选项 1：使用 uv

本节将指导您使用 `uv` 的 `uv pip` 接口进行 Python 设置和包安装过程。对于大多数使用过 pip 的 Python 用户来说，`uv pip` 接口可能比原生的 `uv` 命令更熟悉。

&nbsp;

> **注意：**
> 有其他安装 Python 和使用 `uv` 的方法。例如，您可以通过 `uv` 直接安装 Python 并使用 `uv add` 而不是 `uv pip install` 以获得更快的包管理。
>
> 如果您是 macOS 或 Linux 用户并且更喜欢原生的 `uv` 命令，请参考 [./native-uv.md 教程](./native-uv.md)。我还建议查看官方的 [`uv` 文档](https://docs.astral.sh/uv/)。
>
> `uv add` 语法也适用于 Windows 用户。但是，我发现 `pyproject.toml` 中的某些依赖项在 Windows 上会引起问题。因此，对于 Windows 用户，我推荐 `pixi`，它具有类似于 `uv add` 的 `pixi add` 工作流程。有关更多信息，请参阅 [./native-pixi.md 教程](./native-pixi.md)。
>
> 虽然 `uv add` 和 `pixi add` 提供了额外的速度优势，但我认为 `uv pip` 略微更用户友好，是初学者很好的起点。但是，如果您是 Python 包管理的新手，原生 `uv` 接口也是从一开始就学习它的好机会。这也是我现在使用 `uv` 的方式，但我意识到如果您来自 `pip` 和 `conda`，入门门槛会稍微高一些。



&nbsp;
## 1. 安装 Python（如果尚未安装）

如果您之前没有在系统上手动安装过 Python，我强烈建议这样做。这有助于防止与操作系统内置 Python 安装可能发生的潜在冲突，这可能导致问题。

但是，即使您之前在系统上安装过 Python，请通过在终端中执行以下代码检查您是否安装了现代版本的 Python（我推荐 3.10 或更高版本）：

```bash
python --version
```
如果返回 3.10 或更高版本，则无需进一步操作。

&nbsp;

> **注意：**
> 如果 `python --version` 表明没有安装 Python 版本，您可能还想检查 `python3 --version`，因为您的系统可能配置为使用 `python3` 命令。

&nbsp;

> **注意：**
> 我建议安装比最新发布版本至少旧 2 个版本的 Python，以确保 PyTorch 兼容性。例如，如果最新版本是 Python 3.13，我建议安装版本 3.10 或 3.11。

否则，如果 Python 未安装或版本较旧，您可以按如下所述为您的操作系统安装它。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/python-not-found.png" width="500" height="auto" alt="未找到 Python">

<br>

**Linux (Ubuntu/Debian)**

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

<br>

**macOS**

如果您使用 Homebrew，请使用以下命令安装 Python：

```bash
brew install python@3.10
```

或者，从官方网站下载并运行安装程序：[https://www.python.org/downloads/](https://www.python.org/downloads/)。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/python-version.png" width="700" height="auto" alt="Python 版本">

<br>

**Windows**

从官方网站下载并运行安装程序：[https://www.python.org/downloads/](https://www.python.org/downloads/)。

&nbsp;

## 2. 创建虚拟环境

我强烈建议在单独的虚拟环境中安装 Python 包，以避免修改操作系统可能依赖的系统级包。要在当前文件夹中创建虚拟环境，请遵循以下三个步骤。

<br>

**1. 安装 uv**

```bash
pip install uv
```

<br>

**2. 创建虚拟环境**

```bash
uv venv --python=python3.10
```

<br>

**3. 激活虚拟环境**

```bash
source .venv/bin/activate
```

&nbsp;

> **注意：**
> 如果您使用的是 Windows，您可能需要将上述命令替换为 `source .venv/Scripts/activate` 或 `.venv/Scripts/activate`。


请注意，每次启动新的终端会话时都需要激活虚拟环境。例如，如果您重新启动终端或计算机，并希望第二天继续处理项目，只需在项目文件夹中运行 `source .venv/bin/activate` 即可重新激活您的虚拟环境。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/venv-activate-1.png" width="600" height="auto" alt="虚拟环境已激活">

您也可以通过执行 `deactivate` 命令来停用环境。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/venv-activate-2.png" width="800" height="auto" alt="虚拟环境已停用">

&nbsp;
## 3. 安装包

激活虚拟环境后，您可以使用 `uv` 安装 Python 包。例如：

```bash
uv pip install packaging
```

要从位于此 GitHub 存储库顶层的 `requirements.txt` 文件安装所有必需的包，请运行以下命令，假设该文件与您的终端会话位于同一目录中：

```bash
uv pip install -r requirements.txt
```


或者直接从存储库安装最新的依赖项：

```bash
uv pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt
```


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/uv-install.png" width="700" height="auto" alt="Uv 安装">

&nbsp;

> **注意：**
> 如果由于某些依赖项（例如，如果您使用的是 Windows）导致上述命令出现问题，您始终可以回退到使用常规 pip：
> `pip install -r requirements.txt`
> 或
> `pip install -U -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`

&nbsp;

> **奖励材料的可选依赖项：**
> 要包含奖励材料中使用的可选依赖项，请从项目根目录安装 `bonus` 依赖项组：
>  `uv pip install --group bonus`
> 如果您不想在稍后检查可选奖励材料时单独安装它们，这很有用。

<br>

**完成设置**

就是这样！您的环境现在应该可以为存储库中的代码运行做好了准备。

可选地，您可以通过在此存储库中执行 `python_environment_check.py` 脚本来运行环境检查：

```bash
python setup/02_installing-python-libraries/python_environment_check.py
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/env-check.png" width="700" height="auto" alt="环境检查">

如果您遇到特定包的问题，请尝试使用以下命令重新安装它们：

```bash
uv pip install packagename
```

（这里的 `packagename` 是一个占位符名称，需要替换为您遇到问题的包名称。）

如果问题仍然存在，请考虑在 GitHub 上 [发起讨论](https://github.com/rasbt/LLMs-from-scratch/discussions) 或处理下面的 *选项 2：使用 Conda* 部分。

<br>

**开始使用代码**

一旦设置完成，您就可以开始使用代码文件了。例如，通过运行启动 [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/)：

```bash
jupyter lab
```

&nbsp;

> **注意：**
> 如果您遇到 jupyter lab 命令的问题，您也可以使用虚拟环境中的完整路径启动它。例如，在 Linux/macOS 上使用 `.venv/bin/jupyter lab`，或在 Windows 上使用 `.venv\Scripts\jupyter-lab`。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/jupyter.png" width="900" height="auto" alt="Uv 安装">

&nbsp;
<br>
<br>
&nbsp;

# 选项 2：使用 Conda



本节将指导您使用通过 [miniforge](https://github.com/conda-forge/miniforge) 的 [`conda`](https://www.google.com/search?client=safari&rls=en&q=conda&ie=UTF-8&oe=UTF-8) 进行 Python 设置和包安装过程。

在本教程中，我使用的是运行 macOS 的计算机，但此工作流程与 Linux 机器相似，也可能适用于其他操作系统。

&nbsp;
## 1. 下载并安装 Miniforge

从 GitHub 存储库 [这里](https://github.com/conda-forge/miniforge) 下载 miniforge。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/download.png" alt="下载" width="600px">

根据您的操作系统，这应该会下载 `.sh`（macOS、Linux）或 `.exe`（Windows）文件。

对于 `.sh` 文件，打开您的命令行终端并执行以下命令

```bash
sh ~/Desktop/Miniforge3-MacOSX-arm64.sh
```

其中 `Desktop/` 是下载 Miniforge 安装程序的文件夹。在您的计算机上，您可能需要将其替换为 `Downloads/`。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/miniforge-install.png" alt="miniforge-install" width="600px">

接下来，逐步执行下载说明，按"Enter"确认。

&nbsp;
## 2. 创建新的虚拟环境

安装成功完成后，我建议创建一个名为 `LLMs` 的新虚拟环境，您可以通过执行以下命令来实现：

```bash
conda create -n LLMs python=3.10
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/new-env.png" alt="new-env" width="600px">

> 许多科学计算库并不立即支持最新版本的 Python。因此，在安装 PyTorch 时，建议使用一个或两个版本较旧的 Python。例如，如果最新的 Python 版本是 3.13，建议使用 Python 3.10 或 3.11。

接下来，激活新的虚拟环境（每次打开新的终端窗口或标签时都必须执行）：

```bash
conda activate LLMs
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/activate-env.png" alt="activate-env" width="600px">


&nbsp;
## 可选：设置您的终端样式

如果您想设置终端样式以类似于我的样式，以便可以看到哪个虚拟环境处于活动状态，请查看 [Oh My Zsh](https://github.com/ohmyzsh/ohmyzsh) 项目。

&nbsp;
## 3. 安装新的 Python 库



要安装新的 Python 库，您现在可以使用 `conda` 包安装程序。例如，您可以安装 [JupyterLab](https://jupyter.org/install) 和 [watermark](https://github.com/rasbt/watermark)，如下所示：

```bash
conda install jupyterlab watermark
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/conda-install.png" alt="conda-install" width="600px">



您仍然可以使用 `pip` 安装库。默认情况下，`pip` 应该链接到您的新的 `LLms` conda 环境：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/check-pip.png" alt="check-pip" width="600px">

&nbsp;
## 4. 安装 PyTorch

PyTorch 可以像安装任何其他 Python 库或包一样使用 pip 安装。例如：

```bash
pip install torch
```

但是，由于 PyTorch 是一个包含 CPU 和 GPU 兼容代码的全面库，安装可能需要额外的设置和说明（更多信息请参见书中的 *A.1.3 安装 PyTorch*）。

还强烈建议在官方 PyTorch 网站上的 [https://pytorch.org](https://pytorch.org) 查看安装指南菜单。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/pytorch-installer.jpg" width="600px">

&nbsp;
## 5. 安装本书中使用的 Python 包和库

有关如何安装所需库的说明，请参阅 [本书中使用的 Python 包和库安装](../02_installing-python-libraries/README.md) 文档。

<br>

---

有任何问题？请在 [讨论论坛](https://github.com/rasbt/LLMs-from-scratch/discussions) 中随时与我们联系。