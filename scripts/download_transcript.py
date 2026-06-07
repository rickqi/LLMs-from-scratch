#!/usr/bin/env python3
"""
YouTube 视频字幕下载 + 中文学术笔记生成工具

用法:
  python download_transcript.py

输出:
  docs/transcripts/<video_id>_en.md     英文原始文稿
  docs/transcripts/<video_id>_cn.md     中文翻译文稿（需配置翻译方式）
"""

import subprocess
import sys
import os
import re
import json
from pathlib import Path
from datetime import timedelta

# ============================================================
# 配置区 —— 在这里添加要下载的视频
# ============================================================
VIDEOS = [
    {
        "url": "https://www.youtube.com/watch?v=yAcWnfsZhzo",
        "title_zh": "Python 环境配置指南",
        "chapter": "第 1 章",
        "note_file": "02-环境配置.md",
    },
    # 添加更多视频:
    # {"url": "https://www.youtube.com/watch?v=XXXXX", "title_zh": "...", "chapter": "第 X 章", "note_file": "XX-xxx.md"},
]

# ============================================================
# 翻译配置（三选一）
# ============================================================
# 方式 1：使用 googletrans（免费，质量一般）
USE_GOOGLE_TRANSLATE = False

# 方式 2：使用 OpenAI API（需 API key，质量好）
USE_OPENAI = False
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# 方式 3：不自动翻译，手动翻译或后续用 LLM 处理
# (默认，生成英文稿 + 提示信息)

# ============================================================
# 路径配置
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
TRANSCRIPTS_DIR = DOCS_DIR / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


def download_subtitles(url: str) -> dict:
    """使用 yt-dlp 下载英文字幕，返回字幕文件路径和视频元信息"""
    video_id = url.split("v=")[-1].split("&")[0]
    output_template = str(TRANSCRIPTS_DIR / f"{video_id}")

    # 获取视频信息
    info_cmd = [
        "yt-dlp", "--dump-json", "--no-playlist", "--skip-download", url
    ]
    result = subprocess.run(info_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ 获取视频信息失败: {result.stderr}")
        return None

    info = json.loads(result.stdout)
    print(f"🎬 视频: {info.get('title', 'Unknown')}")
    print(f"⏱️  时长: {timedelta(seconds=info.get('duration', 0))}")

    # 下载英文字幕（优先手动字幕，其次自动生成）
    sub_cmd = [
        "yt-dlp",
        "--write-auto-subs",       # 自动生成字幕
        "--sub-langs", "en",       # 英文
        "--convert-subs", "vtt",   # 转为 VTT 格式
        "--skip-download",         # 不下载视频
        "--output", output_template,
        "--no-playlist",
        url,
    ]

    print("📥 正在下载英文字幕...")
    result = subprocess.run(sub_cmd, capture_output=True, text=True)

    # 查找下载的字幕文件
    vtt_file = TRANSCRIPTS_DIR / f"{video_id}.en.vtt"
    if not vtt_file.exists():
        # 尝试不带 language code 的格式
        for f in TRANSCRIPTS_DIR.glob(f"{video_id}*.vtt"):
            vtt_file = f
            break

    if not vtt_file.exists():
        print("❌ 字幕下载失败（该视频可能没有英文字幕）")
        return None

    print(f"✅ 字幕已保存: {vtt_file}")
    return {
        "vtt_path": vtt_file,
        "title": info.get("title", ""),
        "duration": info.get("duration", 0),
        "video_id": video_id,
    }


def parse_vtt(vtt_path: Path) -> list[dict]:
    """解析 VTT 字幕文件，返回 [{start, end, text}, ...]"""
    content = vtt_path.read_text(encoding="utf-8")

    # 移除 WEBVTT 头部
    lines = content.split("\n")
    # 跳过头部
    start_idx = 0
    for i, line in enumerate(lines):
        if "-->" in line:
            start_idx = max(0, i - 1)
            break

    segments = []
    i = start_idx
    while i < len(lines):
        line = lines[i].strip()

        # 跳过序号行和空行
        if not line or line.isdigit():
            i += 1
            continue

        # 时间轴行: 00:00:00.000 --> 00:00:05.000
        if "-->" in line:
            times = line.split("-->")
            start_time = times[0].strip()
            end_time = times[1].strip().split(" ")[0]  # 去掉可能的样式标记

            # 收集文本（可能跨多行，直到空行或下一个时间轴）
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                text = lines[i].strip()
                # 去除 VTT 标签
                text = re.sub(r"<[^>]+>", "", text)
                if text:
                    text_lines.append(text)
                i += 1

            segments.append({
                "start": start_time,
                "end": end_time,
                "text": " ".join(text_lines),
            })
            continue

        i += 1

    return segments


def format_timestamp(ts: str) -> str:
    """将 VTT 时间戳转为可读格式，如 00:01:23"""
    parts = ts.replace(",", ".").split(":")
    if len(parts) == 3:
        h, m, s = parts
        return f"{int(h):02d}:{int(m):02d}:{int(float(s)):02d}"
    return ts


def segments_to_markdown(segments: list[dict], meta: dict, title_zh: str,
                         chapter: str) -> str:
    """将字幕分段转成 Markdown 学习笔记"""
    lines = []
    lines.append(f"# {chapter}：{title_zh}")
    lines.append("")
    lines.append(f"> 🎬 来源视频: [{meta['title']}](https://youtube.com/watch?v={meta['video_id']})")
    lines.append(f"> ⏱️ 总时长: {timedelta(seconds=meta['duration'])}")
    lines.append(f"> 📅 生成日期: 自动生成")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 视频文稿（英文原文 + 时间戳）")
    lines.append("")

    # 合并连续短句，按时间分组
    current_text = []
    current_start = None
    current_end = None

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue

        if current_start is None:
            current_start = seg["start"]
            current_end = seg["end"]
            current_text = [text]
        elif len(" ".join(current_text)) + len(text) < 500:
            # 合并连续段落（<500 字符）
            current_text.append(text)
            current_end = seg["end"]
        else:
            # 输出当前组
            ts = format_timestamp(current_start)
            lines.append(f"**[{ts}]**  ")
            lines.append(" ".join(current_text))
            lines.append("")
            current_start = seg["start"]
            current_end = seg["end"]
            current_text = [text]

    # 输出最后一组
    if current_text:
        ts = format_timestamp(current_start)
        lines.append(f"**[{ts}]**  ")
        lines.append(" ".join(current_text))
        lines.append("")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 关键要点（待补充）")
    lines.append("")
    lines.append("> 观看视频后，在此记录核心知识点：")
    lines.append("")
    lines.append("- [ ] 要点 1")
    lines.append("- [ ] 要点 2")
    lines.append("- [ ] 要点 3")
    lines.append("")

    return "\n".join(lines)


def translate_to_chinese(text: str) -> str:
    """翻译英文文本到中文"""
    if USE_OPENAI and OPENAI_API_KEY:
        return _translate_openai(text)
    elif USE_GOOGLE_TRANSLATE:
        return _translate_google(text)
    else:
        return text  # 不翻译


def _translate_openai(text: str) -> str:
    """使用 OpenAI API 翻译"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # 分段翻译（避免超出 token 限制）
        paragraphs = text.split("\n\n")
        translated = []

        for para in paragraphs:
            if not para.strip() or para.startswith("**[") or para.startswith("#"):
                translated.append(para)
                continue

            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "你是专业的技术翻译。将以下英文翻译为简体中文，保留技术术语的英文原文（如 PyTorch、pip、conda）。保持 Markdown 格式。"},
                    {"role": "user", "content": para},
                ],
                temperature=0.1,
            )
            translated.append(resp.choices[0].message.content)

        return "\n\n".join(translated)
    except Exception as e:
        print(f"⚠️ OpenAI 翻译失败: {e}")
        return text


def _translate_google(text: str) -> str:
    """使用 Google Translate"""
    try:
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, dest="zh-cn")
        return result.text
    except ImportError:
        print("⚠️ 请先安装: pip install googletrans==4.0.0rc1")
        return text
    except Exception as e:
        print(f"⚠️ Google 翻译失败: {e}")
        return text


def save_note(content: str, filename: str, suffix: str = ""):
    """保存学习笔记"""
    if suffix:
        name, ext = os.path.splitext(filename)
        filename = f"{name}{suffix}{ext}"

    filepath = DOCS_DIR / filename
    filepath.write_text(content, encoding="utf-8")
    print(f"📝 笔记已保存: {filepath}")
    return filepath


def main():
    print("=" * 50)
    print("  YouTube → 中文学习笔记 转换工具")
    print("=" * 50)
    print()

    for video in VIDEOS:
        url = video["url"]
        title_zh = video["title_zh"]
        chapter = video.get("chapter", "")
        note_file = video.get("note_file", "")

        print(f"\n{'='*40}")
        print(f"  处理: {title_zh}")
        print(f"{'='*40}")

        # 1. 下载字幕
        meta = download_subtitles(url)
        if not meta:
            print(f"  ⏭️  跳过: {title_zh}")
            continue

        # 2. 解析字幕
        print("📖 解析字幕...")
        segments = parse_vtt(meta["vtt_path"])
        print(f"   共 {len(segments)} 个分段")

        if not segments:
            print("  ⚠️ 未提取到有效字幕内容")
            continue

        # 3. 生成英文 Markdown
        en_content = segments_to_markdown(segments, meta, title_zh, chapter)
        if note_file:
            save_note(en_content, note_file, "_en")

        # 4. 翻译为中文（如已配置）
        if USE_OPENAI or USE_GOOGLE_TRANSLATE:
            print("🌐 翻译中...")
            cn_content = translate_to_chinese(en_content)
            if note_file:
                save_note(cn_content, note_file)
        elif note_file:
            # 不翻译：直接保存英文版作为主文件
            save_note(en_content, note_file)
            print()
            print("  💡 提示：英文文稿已保存。如需中文翻译：")
            print("     方式1: 设置 USE_OPENAI=True 并配置 OPENAI_API_KEY")
            print("     方式2: 设置 USE_GOOGLE_TRANSLATE=True")
            print("     方式3: 将 *.md 文件贴给 ChatGPT/DeepSeek 翻译")

        # 5. 清理临时 VTT 文件
        if meta["vtt_path"].exists():
            meta["vtt_path"].unlink()
            print(f"🧹 已清理临时文件: {meta['vtt_path'].name}")

    print(f"\n{'='*50}")
    print("✅ 全部完成！笔记保存在 docs/ 目录下")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
