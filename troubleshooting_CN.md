# 常见问题排查指南

本页汇总了在学习本书过程中遇到的常见问题和环境配置技巧。

## Notebook 图片加载问题

各章节的 Notebook 使用托管在 `https://sebastianraschka.com/images/LLMs-from-scratch-images/...` 的 Markdown 图片链接。这样做可以保持仓库下载体积较小，但同时也意味着图片加载依赖于图片托管服务器和你的网络连接。

如果 `.ipynb` Notebook 中的图片无法渲染：

- 在浏览器中直接打开其中一个图片 URL，例如 [https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp)。
- 如果该 URL 在浏览器中同样无法加载，问题很可能是临时的网站、DNS、VPN、代理、防火墙或本地网络问题，而非 Notebook 本身的问题。
- 建议在其他设备或网络上再次检查该 URL（例如，尝试用手机打开图片）；如果图片在手机上可以正常加载，则问题可能指向你电脑上的 VPN 或防火墙配置。
- 如果图片在手机上也无法加载，请随时在 GitHub 上提交 [Issue](https://github.com/rasbt/LLMs-from-scratch/issues) 来帮助我进一步排查。

## 在更新仓库的同时保留个人 Notebook 修改

如果你想修改 Notebook 同时又能接收仓库更新，请先 Fork 本仓库，然后克隆你 Fork 的版本。主要的书籍 Notebook 与纸质书保持一致，一般不会更改（除关键修复外）。大多数仓库更新是新增补充材料。

Notebook 文件本质上是 JSON 文件，因此 Git diff 和合并冲突可能难以阅读。为避免不必要的冲突，建议将你的实验代码与受版本控制的书籍 Notebook 分开存放：

- 修改前先复制 Notebook，例如从 `ch02.ipynb` 复制为 `ch02_experiments.ipynb`。
- 将你的实验 Notebook 放在单独的文件夹或自己的分支上。
- 通过 `upstream` 远程仓库拉取原始仓库的更新，只在需要这些更新时才合并或变基。

创建 Fork 并克隆的步骤：

1. 打开 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)。
2. 点击 GitHub 页面右上角的 **Fork** 按钮。
3. 克隆你的 Fork，将 `YOUR-USERNAME` 替换为你的 GitHub 用户名：

```bash
git clone https://github.com/YOUR-USERNAME/LLMs-from-scratch.git
cd LLMs-from-scratch
```

然后将原始仓库添加为 `upstream` 远程仓库，以便获取未来更新：

```bash
git remote add upstream https://github.com/rasbt/LLMs-from-scratch.git
git fetch upstream
git merge upstream/main
```

如果你确实需要合并已修改的 Notebook，建议安装 [`nbdime`](https://nbdime.readthedocs.io/) 来获取 Notebook 友好的 diff 和合并工具：

```bash
pip install nbdime
nbdime config-git --enable
```

更多背景信息参见 [#1015](https://github.com/rasbt/LLMs-from-scratch/issues/1015)。

## Apple Silicon 和 MPS 支持

部分 Notebook 和脚本在可用时使用 `cuda`，否则回退到 `cpu`，而没有选择 Apple 的 `mps` 后端。在许多地方有意省略 `mps` 支持，因为早期的 PyTorch/MPS 版本在多个示例中（尤其是在训练和微调期间）产生了不稳定或不同的结果。

如果你使用的是 Apple Silicon Mac 并看到损失值发散、损失剧烈波动、生成文本质量差或结果与书中不一致的情况，请先尝试在 `cpu` 上重新运行示例。为了获得与书中一致的训练速度，建议在本地 NVIDIA GPU 或云 GPU 上使用 `cuda`。

较新的 PyTorch 版本可能会改善 MPS 的行为，你可以在本地尝试 `mps`，前提是仔细验证结果。但如果你自己为脚本添加 `mps` 支持，请注意 CUDA 特定的选项（如 `pin_memory=True`、`torch.compile` 和 DDP/多 GPU 代码）可能需要单独的处理。

更多背景信息参见 [#977](https://github.com/rasbt/LLMs-from-scratch/issues/977)、[#625](https://github.com/rasbt/LLMs-from-scratch/discussions/625)、[#644](https://github.com/rasbt/LLMs-from-scratch/discussions/644)、[#442](https://github.com/rasbt/LLMs-from-scratch/discussions/442) 和 [#846](https://github.com/rasbt/LLMs-from-scratch/issues/846)。

## 其他问题

对于其他问题，请随时在 GitHub 上提交新的 [Issue](https://github.com/rasbt/LLMs-from-scratch/issues)。
