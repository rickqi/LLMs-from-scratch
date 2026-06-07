# 《Build a Large Language Model (From Scratch)》中文学习资源

## 视频索引

| 文件 | 章节 | 标题 | 字幕 | 视频 | YouTube |
|------|------|------|------|------|---------|
| 02 | 第 1 章 | Python 环境配置指南 | [02-环境配置.md](youtube/02-环境配置.md) | `02-环境配置.mp4` | [▶](https://www.youtube.com/watch?v=yAcWnfsZhzo) |
| 03 | 第 1 章 | LLM 开发生命周期概述 | [03-LLM生命周期.md](youtube/03-LLM生命周期.md) | `03-LLM生命周期.mp4` | [▶](https://www.youtube.com/watch?v=kPGTx4wcm_w) |
| 04 | 第 2 章 | 文本数据处理跟练 | [04-文本数据处理.md](youtube/04-文本数据处理.md) | `04-文本数据处理.mp4` | [▶](https://www.youtube.com/watch?v=341Rb8fJxY0) |
| 05 | 第 3 章 | 注意力机制跟练 | [05-注意力机制.md](youtube/05-注意力机制.md) | `05-注意力机制.mp4` | [▶](https://www.youtube.com/watch?v=-Ll8DtpNtvk) |
| 06 | 第 3 章 | PyTorch Buffers 深入理解 | [06-PyTorch-Buffers.md](youtube/06-PyTorch-Buffers.md) | `06-PyTorch-Buffers.mp4` | [▶](https://www.youtube.com/watch?v=PetlIokI9Ao) |
| 07 | 第 4 章 | GPT 模型实现跟练 | [07-GPT模型实现.md](youtube/07-GPT模型实现.md) | `07-GPT模型实现.mp4` | [▶](https://www.youtube.com/watch?v=YSAkgEarBGE) |
| 08 | 第 5 章 | 预训练跟练 | [08-预训练.md](youtube/08-预训练.md) | `08-预训练.mp4` | [▶](https://www.youtube.com/watch?v=Zar2TJv-sE0) |
| 09 | 第 6 章 | 分类微调跟练 | [09-分类微调.md](youtube/09-分类微调.md) | `09-分类微调.mp4` | [▶](https://www.youtube.com/watch?v=5PFXJYme4ik) |
| 10 | 第 7 章 | 指令微调跟练 | [10-指令微调.md](youtube/10-指令微调.md) | `10-指令微调.mp4` | [▶](https://www.youtube.com/watch?v=4yNswvhPWCQ) |

## 学习笔记

| 文件 | 内容 |
|------|------|
| [01-学习路线与建议.md](01-学习路线与建议.md) | 全书学习路线、四阶段规划、25 个考试卡片 |
| [02-环境配置.md](02-环境配置.md) | Python 环境配置指南（UV、虚拟环境、依赖安装） |
| [03-LLM生命周期与第2章预习.md](03-LLM生命周期与第2章预习.md) | LLM 开发生命周期 + 第 2 章文本数据处理预习 |
| [04-注意力机制.md](04-注意力机制.md) | 注意力机制（第 3 章）：Simple → Self → Causal → Multi-Head |
| [05-GPT模型实现.md](05-GPT模型实现.md) | GPT 模型实现（第 4 章）：LayerNorm、GELU、TransformerBlock、GPT-2 架构 |
| [06-预训练.md](06-预训练.md) | 预训练（第 5 章）：训练循环、文本生成、加载预训练权重 |
| [07-分类微调.md](07-分类微调.md) | 分类微调（第 6 章）：替换输出头、冻结策略、垃圾分类实战 |
| [08-指令微调.md](08-指令微调.md) | 指令微调（第 7 章）：Alpaca 格式、response-only loss、DPO 对齐 |
| [09-PyTorch入门.md](09-PyTorch入门.md) | PyTorch 入门（附录 A）：Tensor、autograd、nn.Module、训练循环、GPU 训练 |
| [10-LoRA参数高效微调.md](10-LoRA参数高效微调.md) | LoRA 参数高效微调（附录 E）：低秩分解、LoRALayer 实战、50x 参数缩减 |

## 目录结构

```
docs/
├── 01-学习路线与建议.md              ← 学习路线总览（含 25 个考试卡片）
├── 02-环境配置.md                    ← 环境配置学习笔记
├── 03-LLM生命周期与第2章预习.md      ← LLM 生命周期 + Ch2 预习
├── 04-注意力机制.md                  ← Ch3：注意力机制学习笔记
├── 05-GPT模型实现.md                 ← Ch4：GPT 模型实现学习笔记
├── 06-预训练.md                      ← Ch5：预训练学习笔记
├── 07-分类微调.md                    ← Ch6：分类微调学习笔记
├── 08-指令微调.md                    ← Ch7：指令微调学习笔记
├── 09-PyTorch入门.md                 ← 附录 A：PyTorch 基础
├── 10-LoRA参数高效微调.md            ← 附录 E：LoRA 参数高效微调
├── README.md                         ← 本文件
└── youtube/
    ├── transcript_to_cn.py           ← 字幕下载脚本
    ├── make_bilingual_srt.py         ← 双语字幕生成脚本
    ├── 02-环境配置.md / .mp4 / .bilingual.srt
    ├── 03-LLM生命周期.md / .mp4 / .bilingual.srt
    ├── 04-文本数据处理.md / .mp4 / .bilingual.srt
    ├── 05-注意力机制.md / .mp4 / .bilingual.srt
    ├── 06-PyTorch-Buffers.md / .mp4 / .bilingual.srt
    ├── 07-GPT模型实现.md / .mp4 / .bilingual.srt
    ├── 08-预训练.md / .mp4 / .bilingual.srt
    ├── 09-分类微调.md / .mp4 / .bilingual.srt
    └── 10-指令微调.md / .mp4 / .bilingual.srt
```

## 重新下载

```bash
cd /path/to/LLMs-from-scratch
source .venv/bin/activate
pip install yt-dlp

# 仅字幕
python docs/youtube/transcript_to_cn.py

# 字幕 + 视频（720p mp4）
python docs/youtube/transcript_to_cn.py --video
```

WSL2 环境下脚本会自动检测 Windows Clash 代理，无需手动配置。

## 字幕翻译

字幕为中英双语格式（DeepSeek 翻译，技术术语保留英文原文）。

双语 `.srt` 文件可用 VLC、mpv 等播放器加载到对应视频中。
