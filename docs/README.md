# YouTube 视频 → 中文学术笔记

## 快速使用

```bash
cd /home/LLMs-from-scratch
source .venv/bin/activate

# 1. 下载字幕 + 生成英文 Markdown 笔记
python scripts/download_transcript.py

# 2. （可选）翻译为中文
#    方式 A: 直接贴给 ChatGPT/DeepSeek，提示词：
#       "将以下英文技术文稿翻译为中文，保留技术术语英文原文"
#    方式 B: 在 download_transcript.py 中设置翻译方式
```

## 输出位置

```
docs/
├── 01-学习路线与建议.md        ← 已生成
├── 02-环境配置.md              ← 🎯 目标（需运行脚本生成）
└── transcripts/                ← 临时字幕文件
```

## 添加更多视频

编辑 `scripts/download_transcript.py`，在 `VIDEOS` 列表中添加：

```python
VIDEOS = [
    # 已有
    {"url": "https://www.youtube.com/watch?v=yAcWnfsZhzo",
     "title_zh": "Python 环境配置指南",
     "chapter": "第 1 章",
     "note_file": "02-环境配置.md"},
    # 新增
    {"url": "https://www.youtube.com/watch?v=XXXXX",
     "title_zh": "你的标题",
     "chapter": "第 X 章",
     "note_file": "XX-篇章名.md"},
]
```

## 注意

当前服务器无法访问 YouTube（HTTPS 阻断），请在本地机器上运行此脚本。
