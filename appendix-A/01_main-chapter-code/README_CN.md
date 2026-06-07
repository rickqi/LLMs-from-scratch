# 附录 A：PyTorch 入门

### 主章节代码

- [code-part1.ipynb](code-part1.ipynb) 包含 A.1 到 A.8 节中出现的所有代码
- [code-part2.ipynb](code-part2.ipynb) 包含 A.9 节 GPU 相关代码
- [DDP-script.py](DDP-script.py) 包含演示多 GPU 使用的脚本（注意 Jupyter Notebook 仅支持单 GPU，因此这是一个脚本而非 Notebook）。可以通过 `python DDP-script.py` 运行。如果你的机器有 2 个以上 GPU，使用 `CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py` 运行。
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章的练习解答

### 可选代码

- [DDP-script-torchrun.py](DDP-script-torchrun.py) 是 `DDP-script.py` 的可选版本，通过 PyTorch 的 `torchrun` 命令运行，而不是通过 `multiprocessing.spawn` 自行生成和管理多个进程。`torchrun` 命令的优点是自动处理分布式初始化（包括多节点协调），略微简化了设置流程。可以通过 `torchrun --nproc_per_node=2 DDP-script-torchrun.py` 使用该脚本。
