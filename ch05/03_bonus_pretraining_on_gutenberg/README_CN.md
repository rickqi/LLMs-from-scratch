# 在 Project Gutenberg 数据集上预训练 GPT

此目录中的代码包含用于在 Project Gutenberg 提供的免费书籍上训练小型 GPT 模型的代码。

正如 Project Gutenberg 网站所述，"Project Gutenberg 电子书中绝大多数在美国属于公共领域"。

请阅读 [Project Gutenberg 权限、许可和其他常见请求](https://www.gutenberg.org/policy/permission.html) 页面，了解有关使用 Project Gutenberg 提供的资源的更多信息。

&nbsp;
## 如何使用此代码

&nbsp;

### 1) 下载数据集

在本节中，我们使用来自 [`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) GitHub 存储库的代码从 Project Gutenberg 下载书籍。

截至撰写本文时，这需要大约 50 GB 的磁盘空间，并花费大约 10-15 小时，但此后可能会根据 Project Gutenberg 的增长而更多。

&nbsp;
#### Linux 和 macOS 用户的下载说明

Linux 和 macOS 用户可以按照以下步骤下载数据集（如果您是 Windows 用户，请参见下面的说明）：

1. 将 `03_bonus_pretraining_on_gutenberg` 文件夹设置为工作目录，以便在此文件夹中本地克隆 `gutenberg` 存储库（这是运行提供的脚本 `prepare_dataset.py` 和 `pretraining_simple.py` 所必需的）。例如，当在 `LLMs-from-scratch` 存储库的文件夹中时，通过以下方式导航到 *03_bonus_pretraining_on_gutenberg* 文件夹：

```bash
cd ch05/03_bonus_pretraining_on_gutenberg
```

2. 在那里克隆 `gutenberg` 存储库：

```bash
git clone https://github.com/pgcorpus/gutenberg.git
```

3. 进入本地克隆的 `gutenberg` 存储库文件夹：

```bash
cd gutenberg
```

4. 从 `gutenberg` 存储库的文件夹中安装 *requirements.txt* 中定义的必需包：

```bash
pip install -r requirements.txt
```

5. 下载数据：

```bash
python get_data.py
```

6. 返回到 `03_bonus_pretraining_on_gutenberg` 文件夹

```bash
cd ..
```

&nbsp;
#### Windows 用户的特殊说明

[`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) 代码与 Linux 和 macOS 兼容。然而，Windows 用户需要进行一些小调整，例如在 `subprocess` 调用中添加 `shell=True` 并替换 `rsync`。

另一种在 Windows 上运行此代码的更简单方法是使用"适用于 Linux 的 Windows 子系统"（WSL）功能，该功能允许用户在 Windows 中使用 Ubuntu 运行 Linux 环境。有关更多信息，请阅读 [Microsoft 官方安装说明](https://learn.microsoft.com/en-us/windows/wsl/install) 和 [教程](https://learn.microsoft.com/en-us/training/modules/wsl-introduction/)。

使用 WSL 时，请确保您已安装 Python 3（通过 `python3 --version` 检查，或例如使用 `sudo apt-get install -y python3.10` 安装 Python 3.10），并在那里安装以下包：

```bash
sudo apt-get update && \
sudo apt-get upgrade -y && \
sudo apt-get install -y python3-pip && \
sudo apt-get install -y python-is-python3 && \
sudo apt-get install -y rsync
```

> **注意：**
> 关于如何设置 Python 和安装包的说明可以在 [可选 Python 设置偏好](../../setup/01_optional-python-setup-preferences/README.md) 和 [安装 Python 库](../../setup/02_installing-python-libraries/README.md) 中找到。
>
> 可选地，此存储库提供了一个运行 Ubuntu 的 Docker 镜像。关于如何使用提供的 Docker 镜像运行容器的说明可以在 [可选 Docker 环境](../../setup/03_optional-docker-environment/README.md) 中找到。

&nbsp;
### 2) 准备数据集

接下来，运行 `prepare_dataset.py` 脚本，该脚本将（截至撰写本文时为 60,173 个）文本文件连接到更少的大文件中，以便更有效地传输和访问：

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

```
...
Skipping gutenberg/data/raw/PG29836_raw.txt as it does not contain primarily English text.                                     Skipping gutenberg/data/raw/PG16527_raw.txt as it does not contain primarily English text.                                     100%|██████████████████████████████████████████████████████████| 57250/57250 [25:04<00:00, 38.05it/s]
42 file(s) saved in /Users/sebastian/Developer/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg/gutenberg_preprocessed
```


> **提示：**
> 注意，生成的文件以纯文本格式存储，为简单起见未预分词。但是，如果您计划更频繁地使用数据集或进行多个 epoch 的训练，您可能希望更新代码以将数据集存储在预分词形式中以节省计算时间。请参阅页面底部的*设计决策和改进*以获取更多信息。

> **提示：**
> 您可以选择更小的文件大小，例如 50 MB。这将产生更多文件，但对于测试目的在少量文件上快速预训练运行可能有用。

&nbsp;
### 3) 运行预训练脚本

您可以如下运行预训练脚本。注意，额外的命令行参数显示的是默认值，用于说明目的：

```bash
python pretraining_simple.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

输出将按以下方式格式化：

> Total files: 3
> Tokenizing file 1 of 3: data_small/combined_1.txt
> Training ...
> Ep 1 (Step 0): Train loss 9.694, Val loss 9.724
> Ep 1 (Step 100): Train loss 6.672, Val loss 6.683
> Ep 1 (Step 200): Train loss 6.543, Val loss 6.434
> Ep 1 (Step 300): Train loss 5.772, Val loss 6.313
> Ep 1 (Step 400): Train loss 5.547, Val loss 6.249
> Ep 1 (Step 500): Train loss 6.182, Val loss 6.155
> Ep 1 (Step 600): Train loss 5.742, Val loss 6.122
> Ep 1 (Step 700): Train loss 6.309, Val loss 5.984
> Ep 1 (Step 800): Train loss 5.435, Val loss 5.975
> Ep 1 (Step 900): Train loss 5.582, Val loss 5.935
> ...
> Ep 1 (Step 31900): Train loss 3.664, Val loss 3.946
> Ep 1 (Step 32000): Train loss 3.493, Val loss 3.939
> Ep 1 (Step 32100): Train loss 3.940, Val loss 3.961
> Saved model_checkpoints/model_pg_32188.pth
> Book processed 3h 46m 55s
> Total time elapsed 3h 46m 55s
> ETA for remaining books: 7h 33m 50s
> Tokenizing file 2 of 3: data_small/combined_2.txt
> Training ...
> Ep 1 (Step 32200): Train loss 2.982, Val loss 4.094
> Ep 1 (Step 32300): Train loss 3.920, Val loss 4.097
> ...
```


&nbsp;
> **提示：**
> 实际上，如果您使用 macOS 或 Linux，我建议使用 `tee` 命令将日志输出保存到 `log.txt` 文件中，除了在终端上打印：

```bash
python -u pretraining_simple.py | tee log.txt
```

&nbsp;
> **警告：**
> 请注意，在 `gutenberg_preprocessed` 文件夹中的 `~500` Mb 文本文件之一上进行训练将在 V100 GPU 上花费大约 4 小时。
> 文件夹包含 47 个文件，将花费大约 200 小时（超过 1 周）才能完成。您可能希望在较少文件上运行它。

&nbsp;
## 设计决策和改进

请注意，此代码重点保持简单和最少，用于教育目的。代码可以通过以下方式改进以提高建模性能和训练效率：

1. 修改 `prepare_dataset.py` 脚本以从每本书文件中剥离 Gutenberg 标准文本。
2. 更新数据准备和加载工具，以预分词数据集并将其存储在分词形式中，这样在调用预训练脚本时就不必重新分词。
3. 通过添加 [附录 D：为训练循环添加花里胡哨功能](../../appendix-D/01_main-chapter-code/appendix-D.ipynb) 中引入的功能，即余弦衰减、线性预热和梯度裁剪，来更新 `train_model_simple` 脚本。
4. 更新预训练脚本以保存优化器状态（参见第 5 章中的 *5.4 在 PyTorch 中加载和保存权重*；[ch05.ipynb](../../ch05/01_main-chapter-code/ch05.ipynb)），并添加加载现有模型和优化器检查点的选项，如果训练运行被中断则继续训练。
5. 添加更高级的记录器（例如 Weights and Biases）以实时查看损失和验证曲线
6. 添加分布式数据并行（DDP）并在多个 GPU 上训练模型（参见附录 A 中的 *A.9.3 使用多个 GPU 训练*；[DDP-script.py](../../appendix-A/01_main-chapter-code/DDP-script.py)）。
7. 将 [从零开始的 `MultiheadAttention` 类](../../ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb) 奖励部分中的高效 `MHAPyTorchScaledDotProduct` 类替换为第 4 章的 `previous_chapter.py` 脚本，该类使用通过 PyTorch 的 `nn.functional.scaled_dot_product_attention` 函数的 Flash Attention。
8. 通过 [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)（`model = torch.compile`）或 [thunder](https://github.com/Lightning-AI/lightning-thunder)（`model = thunder.jit(model)`）优化模型来加速训练。
9. 实现梯度低秩投影（GaLore）以进一步加速预训练过程。这可以通过将 `AdamW` 优化器替换为 [GaLore Python 库](https://github.com/jiaweizzhao/GaLore) 中提供的 `GaLoreAdamW` 来实现。