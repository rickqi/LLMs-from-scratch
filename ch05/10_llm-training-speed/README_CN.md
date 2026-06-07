# 更快的 LLM 训练的 PyTorch 性能技巧


请注意，本书是为教育目的而编写的，这意味着原始代码特意保持简单。这有助于提高可读性，并确保在不同硬件（包括 CPU 和 GPU）上的兼容性。但是，您可能对一些更高级的 PyTorch 和 GPU 功能感兴趣，以使 LLM 训练性能更高。

此文件夹包含三个代码文件，演示了第 5 章介绍的 LLM 和训练函数的性能优化：

1. [`00_orig.py`](00_orig.py): 第 5 章的原始代码，用于 CPU 和单 GPU 训练。  
   ➤ 通过以下方式运行：`python 00_orig.py`

2. [`01_opt_single_gpu.py`](01_opt_single_gpu.py): 单 GPU 训练的优化版本。  
   ➤ 通过以下方式运行：`python 01_opt_single_gpu.py`

3. [`02_opt_multi_gpu_ddp.py`](02_opt_multi_gpu_ddp.py): 使用分布式数据并行（DDP）进行多 GPU 训练的优化版本。  
   ➤ 通过以下方式运行：`torchrun --nproc_per_node=4 02_opt_multi_gpu_ddp.py`  
   （**注意**：为了与 `01_opt_single_gpu.py` 保持最小的更改，此脚本仅通过上面所示的 `torchrun` 支持多处理。这意味着多 GPU 支持**不**通过 `python 02_opt_multi_gpu_ddp.py` 支持）

**请注意，这些修改将训练速度从 12,525 tokens/秒（单个 A100）提升到 142,156 tokens/秒（单个 A100）和 419,259 tokens/秒（4x A100）。**

我计划在未来更详细地阐述这些差异。目前，查看代码中添加了哪些改进的最简单方法是使用 Visual Studio Code 打开文件，并通过"比较选定"功能查看差异。

![VS compare](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/llm-training-speed/vs-code-compare.png)

![PyTorch Tips](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/pytorch-tips/pytorch-tips.webp?1)


&nbsp;
## 单 GPU 速度比较

如上所述，我计划在未来详细阐述这些更改。目前，本节包含一个简单的性能概述，以每个修改的 tokens/秒为单位。所有实验都在 A100 GPU 上运行。

&nbsp;
### 基准线

请注意，`00_orig.py` 作为基准线，不包含重大修改，除了以下内容外，使用第 5 章的代码：

- 4 倍更大的上下文长度（这解释了 `00_orig.py` 相比第 5 章相对较大的内存占用）；
- 4 倍批次大小变化（`00_orig.py` 相对较大内存占用的另一个贡献因素）；
- 更大的公共领域书籍以增加训练数据大小。

超参数不太适合最小化损失和减少过拟合，并且 LLM 在最后生成的文本可能不太高级；然而，这应该不重要，因为主要的要点是这里作为速度参考的 `tok/sec` 指标（越高越好）。

```bash
ubuntu@159-13-52-60:~$ python 00_orig.py
PyTorch version: 2.6.0+cu124
Using cuda
CUDA version: 12.4

Ep 1, Step 000000, Train: 9.535, Val: 9.609, Step tok/sec: 7238, Avg tok/sec: 0
Ep 1, Step 000015, Train: 6.201, Val: 6.152, Step tok/sec: 12545, Avg tok/sec: 12545
Ep 1, Step 000030, Train: 5.663, Val: 5.688, Step tok/sec: 12490, Avg tok/sec: 12517
Ep 1, Step 000045, Train: 5.316, Val: 5.362, Step tok/sec: 12541, Avg tok/sec: 12525
Every effort moves you, and's, and I am not be a

...

Ep 15, Step 000735, Train: 0.227, Val: 6.818, Step tok/sec: 11599, Avg tok/sec: 12248
Ep 15, Step 000750, Train: 0.300, Val: 6.895, Step tok/sec: 12530, Avg tok/sec: 12253
Ep 15, Step 000765, Train: 0.150, Val: 6.914, Step tok/sec: 12532, Avg tok/sec: 12259
Every effort moves you like best to think which he held in the room in him, the interest was the night, the realities of the affairs Bulstrode's duty, now!' the fact is another man, conquests

Allocated memory: 2.5069 GB
Reserved memory: 26.2617 GB
```

请注意，`01_opt_single_gpu.py` 包含下面列出的所有修改。

比较总是基于上一节第一个周期之后的平均 tok/sec 和分配的内存。

&nbsp;
### 1. 动态创建因果掩码

- 不是保存因果掩码，而是动态创建因果掩码以减少内存使用（在这里效果最小，但在像 Llama 3.2 这样支持 131k 输入 tokens 的长上下文模型中可能会累积）

修改前：
- `Avg tok/sec: 12525`
- `Reserved memory: 26.2617 GB`

修改后：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 2. 使用张量核心

- 使用张量核心（仅适用于像 A100 及更新版本的 Ampere GPU）

修改前：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

修改后：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 3. 融合 AdamW 优化器

- 通过设置 `fused=True` 使用 `AdamW` 的融合内核

修改前：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

修改后：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 4. 数据加载器中的固定内存

- 在数据加载器中使用 `pin_memory=True` 来预分配和重用 GPU 内存

修改前：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

修改后：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 5. 使用 bfloat16 精度

- 从 32 位浮点数切换到 16 位大脑浮点数（bfloat16）精度（有关此主题的更多信息，请参阅我的 [文章](https://magazine.sebastianraschka.com/p/the-missing-bits-llama-2-weights)）

修改前：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

修改后：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

&nbsp;
### 6. 使用 PyTorch 类替换从头编写的代码

- 使用 PyTorch 的原生实现替换 LayerNorm 和 GeLU 的从头编写实现

修改前：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

修改后：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

&nbsp;
### 7. 使用 FlashAttention

- 使用 PyTorch 的自注意力函数和 FlashAttention，而不是我们从头编写多头注意力实现。

修改前：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

修改后：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

&nbsp;
### 8. 使用 `pytorch.compile`

- 使用 `torch.compile(model)`。请注意，第一次迭代总是很慢，然后才开始加速。由于 `Avg tok/sec` 测量仅包含平均计算的第一行，我们现在使用第一周期结束时的 `Step tok/sec`。

修改前：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

修改后：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

<br>

---

**Windows 说明**

- 在 Windows 上编译可能很棘手
- `torch.compile()` 使用 Inductor，它会 JIT 编译内核，并且需要工作的 C/C++ 工具链
- 对于 CUDA，Inductor 还依赖于 Triton，可通过社区包 `triton-windows` 获得
  - 如果您看到 `cl not found`，[安装带有"C++ 工作负载"的 Visual Studio Build Tools](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170) 并从"x64 Native Tools"提示运行 Python
  - 如果您看到 `triton not found` 与 CUDA，请安装 `triton-windows`（例如 `uv pip install "triton-windows<3.4"`）
- 对于 CPU，读者进一步建议遵循此 [PyTorch Inductor Windows 指南](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)
  - 这里，安装 Visual Studio 2022 时安装英语语言包以避免 UTF-8 错误很重要
  - 还请注意，代码需要通过"Visual Studio 2022 开发人员命令提示符"运行，而不是笔记本
- 如果此设置证明很棘手，您可以跳过编译；**编译是可选的，所有代码示例在没有它的情况下都能正常工作**

---

&nbsp;
### 9. 词汇填充

- 在这里，我们将词汇大小从 50,257 略微增加到 50,304，这是最接近的 64 的倍数。这个技巧是我前同事 Carlos Mocholi 建议给我的，他提到它最初来自 Andrej Karpathy（可能来自 [此帖子](https://x.com/karpathy/status/1621578354024677377)）。Karpathy 的建议基于与 PyTorch 团队的互动，他们给出了关于 `torch.compile` 的建议，如 [Bertrand Maher](https://www.linkedin.com/feed/update/urn:li:activity:7309569006057795584?commentUrn=urn%3Ali%3Acomment%3A%28activity%3A7309569006057795584%2C7309754284185669632%29&dashCommentUrn=urn%3Ali%3Afsd_comment%3A%287309754284185669632%2Curn%3Ali%3Aactivity%3A7309569006057795584%29) 所述。这个的好资源是 [NVIDIA 关于张量形状的指南](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape)，其中批次大小和线性层维度通常选择为某些值的倍数。此外，词汇填充技巧很久以前就被 NVIDIA 的 Megatron 团队描述过（请参阅 2019 年的 [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) 论文）。

修改前：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

修改后：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

&nbsp;
### 10. 增加批次大小

- 最后，我们将批次大小增加到 GPU 支持的最大 2 的幂

修改前：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

修改后：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`


&nbsp;
## 多 GPU 速度比较

这可能不是完全公平的比较，因为我们现在使用 4 个 GPU 而不是 1 个，但使用分布式数据并行（这是在训练不受有限 GPU 内存瓶颈时可以使用的最快技术），当然可以导致明显的速度提升：

修改前（单 GPU）：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`

修改后（4 GPU）：
- `Step tok/sec: 419259`
- `Reserved memory: 22.7969 GB`