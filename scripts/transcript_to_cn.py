#!/usr/bin/env python3
"""
一键：下载全部配套 YouTube 视频字幕 → 保存学习笔记

用法（在你的本地机器上）：
  cd /path/to/LLMs-from-scratch
  source .venv/bin/activate
  pip install yt-dlp
  python scripts/transcript_to_cn.py

然后复制生成的 .md 文件粘贴到 ChatGPT/DeepSeek 翻译：
  "翻译以下英文技术文稿为简体中文，保留技术术语英文原文"
"""

import subprocess, re, sys
from pathlib import Path

VIDEOS = [
    {"id": "yAcWnfsZhzo",    "chapter": "第 1 章", "title": "Python 环境配置指南",           "file": "02-环境配置.md"},
    {"id": "kPGTx4wcm_w",    "chapter": "第 1 章", "title": "LLM 开发生命周期概述",        "file": "03-LLM生命周期.md"},
    {"id": "341Rb8fJxY0",    "chapter": "第 2 章", "title": "文本数据处理跟练",             "file": "04-文本数据处理.md"},
    {"id": "-Ll8DtpNtvk",    "chapter": "第 3 章", "title": "注意力机制跟练",              "file": "05-注意力机制.md"},
    {"id": "PetlIokI9Ao",    "chapter": "第 3 章", "title": "PyTorch Buffers 深入理解",      "file": "06-PyTorch-Buffers.md"},
    {"id": "YSAkgEarBGE",    "chapter": "第 4 章", "title": "GPT 模型实现跟练",             "file": "07-GPT模型实现.md"},
    {"id": "Zar2TJv-sE0",    "chapter": "第 5 章", "title": "预训练跟练",                  "file": "08-预训练.md"},
    {"id": "5PFXJYme4ik",    "chapter": "第 6 章", "title": "分类微调跟练",                "file": "09-分类微调.md"},
    {"id": "4yNswvhPWCQ",    "chapter": "第 7 章", "title": "指令微调跟练",                "file": "10-指令微调.md"},
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

def download_vtt(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    output = str(DOCS_DIR / f"_{video_id}")
    subprocess.run([
        "yt-dlp", "--no-update",
        "--write-auto-subs", "--sub-langs", "en",
        "--convert-subs", "vtt", "--skip-download",
        "-o", output, url
    ], check=True, capture_output=True)
    vtt = next(DOCS_DIR.glob(f"_{video_id}*.vtt"), None)
    if not vtt:
        print(f"  ⚠️ 字幕未找到: {video_id}")
        return None
    return vtt

def parse_vtt(path):
    text = path.read_text(encoding="utf-8")
    segments = []
    for line in text.split("\n"):
        line = line.strip()
        if "-->" in line:
            segments.append({"time": line.split("-->")[0].strip(), "text": ""})
        elif segments and line and not line.startswith("WEBVTT") and not line.isdigit():
            clean = re.sub(r"<[^>]+>", "", line)
            if clean:
                segments[-1]["text"] += " " + clean
    return [s for s in segments if s["text"].strip()]

def to_markdown(segments, video):
    lines = [
        f"# {video['chapter']}：{video['title']}（视频笔记）",
        "",
        f"> 🎬 [原视频](https://www.youtube.com/watch?v={video['id']})",
        f"> 📅 自动生成 | ⚠️ 英文原文，需手动翻译为中文",
        "",
        "---",
        "",
        "## 英文原文（带时间戳）",
        "",
    ]
    for s in segments:
        ts = s["time"].replace(",", ".").split(":")
        ts_str = f"{int(ts[0]):02d}:{int(ts[1]):02d}:{int(float(ts[2])):02d}"
        lines.append(f"**[{ts_str}]** {s['text'].strip()}")
        lines.append("")
    lines.extend(["", "---", "", "## 📝 中文翻译", "", "> 💡 将以上英文内容粘贴到 ChatGPT/DeepSeek 翻译", "", "## 🎯 关键要点", "", "- [ ] 要点 1", "- [ ] 要点 2", "- [ ] 要点 3"])
    return "\n".join(lines)

success = 0
for v in VIDEOS:
    print(f"\n{'='*50}\n  {v['chapter']}: {v['title']}\n{'='*50}")
    try:
        vtt = download_vtt(v["id"])
        if not vtt:
            continue
        segs = parse_vtt(vtt)
        print(f"  📝 解析到 {len(segs)} 个分段")
        if not segs:
            print("  ⚠️ 无有效字幕内容")
            continue
        md = to_markdown(segs, v)
        path = DOCS_DIR / v["file"]
        path.write_text(md, encoding="utf-8")
        print(f"  💾 已保存: {path}")
        vtt.unlink()
        success += 1
    except Exception as e:
        print(f"  ❌ 失败: {e}")

print(f"\n{'='*50}")
print(f"✅ 完成！成功 {success}/{len(VIDEOS)} 个视频")
print(f"\n📂 文件保存在: {DOCS_DIR}")
print(f"🌐 翻译步骤：将生成的 .md 文件粘贴到 ChatGPT/DeepSeek")
print(f"   提示词: '翻译为简体中文，保留技术术语英文原文'")
print(f"{'='*50}")
