#!/usr/bin/env python3
"""
下载配套 YouTube 视频及字幕 → 保存学习笔记

用法：
  cd /path/to/LLMs-from-scratch
  source .venv/bin/activate
  pip install yt-dlp
  python YouTube/transcript_to_cn.py [--video]

  --video  同时下载视频文件（默认仅下载字幕）
"""

import subprocess, re, sys, os
from pathlib import Path

DOWNLOAD_VIDEO = "--video" in sys.argv

VIDEOS = [
    {"id": "yAcWnfsZhzo",    "chapter": "第 1 章", "title": "Python 环境配置指南",           "file": "02-环境配置.md",        "video": "02-环境配置.mp4"},
    {"id": "kPGTx4wcm_w",    "chapter": "第 1 章", "title": "LLM 开发生命周期概述",        "file": "03-LLM生命周期.md",     "video": "03-LLM生命周期.mp4"},
    {"id": "341Rb8fJxY0",    "chapter": "第 2 章", "title": "文本数据处理跟练",             "file": "04-文本数据处理.md",    "video": "04-文本数据处理.mp4"},
    {"id": "-Ll8DtpNtvk",    "chapter": "第 3 章", "title": "注意力机制跟练",              "file": "05-注意力机制.md",     "video": "05-注意力机制.mp4"},
    {"id": "PetlIokI9Ao",    "chapter": "第 3 章", "title": "PyTorch Buffers 深入理解",      "file": "06-PyTorch-Buffers.md", "video": "06-PyTorch-Buffers.mp4"},
    {"id": "YSAkgEarBGE",    "chapter": "第 4 章", "title": "GPT 模型实现跟练",             "file": "07-GPT模型实现.md",    "video": "07-GPT模型实现.mp4"},
    {"id": "Zar2TJv-sE0",    "chapter": "第 5 章", "title": "预训练跟练",                  "file": "08-预训练.md",         "video": "08-预训练.mp4"},
    {"id": "5PFXJYme4ik",    "chapter": "第 6 章", "title": "分类微调跟练",                "file": "09-分类微调.md",       "video": "09-分类微调.mp4"},
    {"id": "4yNswvhPWCQ",    "chapter": "第 7 章", "title": "指令微调跟练",                "file": "10-指令微调.md",       "video": "10-指令微调.mp4"},
]

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR.mkdir(exist_ok=True)

def get_proxy():
    """Auto-detect WSL proxy (Windows Clash) or use env var."""
    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if proxy:
        return proxy
    # WSL2: detect Windows gateway as Clash proxy
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True, text=True, check=True
        )
        gateway = result.stdout.split()[2]
        return f"http://{gateway}:7890"
    except Exception:
        return None

PROXY = get_proxy()

def download_vtt(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    output = str(SCRIPT_DIR / f"_{video_id}")
    cmd = [
        "yt-dlp", "--no-update",
        "--write-auto-subs", "--sub-langs", "en",
        "--convert-subs", "vtt", "--skip-download",
        "-o", output, url
    ]
    if PROXY:
        cmd.extend(["--proxy", PROXY])
    subprocess.run(cmd, check=True, capture_output=True)
    vtt = next(SCRIPT_DIR.glob(f"_{video_id}*.vtt"), None)
    if not vtt:
        print(f"  ⚠️ 字幕未找到: {video_id}")
        return None
    return vtt

def download_video(video):
    target = SCRIPT_DIR / video["video"]
    if target.exists():
        print(f"  ⏭️ 视频已存在，跳过: {target.name}")
        return True
    url = f"https://www.youtube.com/watch?v={video['id']}"
    cmd = [
        "yt-dlp", "--no-update",
        "-f", "bv*[height<=720]+ba/b[height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", str(target),
        url
    ]
    if PROXY:
        cmd.extend(["--proxy", PROXY])
    print(f"  ⬇️ 下载视频: {video['video']} ...")
    subprocess.run(cmd, check=True)
    return True

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
        path = SCRIPT_DIR / v["file"]
        path.write_text(md, encoding="utf-8")
        print(f"  💾 已保存: {path}")
        vtt.unlink()
        if DOWNLOAD_VIDEO:
            download_video(v)
        success += 1
    except Exception as e:
        print(f"  ❌ 失败: {e}")

print(f"\n{'='*50}")
print(f"✅ 完成！成功 {success}/{len(VIDEOS)} 个视频")
print(f"\n📂 文件保存在: {SCRIPT_DIR}")
print(f"🌐 翻译步骤：将生成的 .md 文件粘贴到 ChatGPT/DeepSeek")
print(f"   提示词: '翻译为简体中文，保留技术术语英文原文'")
print(f"{'='*50}")
