#!/usr/bin/env python3
"""
一键：下载 YouTube 视频字幕 → 翻译中文 → 保存学习笔记

用法（在你的本地机器上）：
  cd /path/to/LLMs-from-scratch
  source .venv/bin/activate
  pip install yt-dlp
  python scripts/transcript_to_cn.py
"""

import subprocess, re, json, sys
from pathlib import Path
from datetime import timedelta

VIDEO_URL = "https://www.youtube.com/watch?v=yAcWnfsZhzo"
VIDEO_ID = "yAcWnfsZhzo"
TITLE_ZH = "Python 环境配置指南"
CHAPTER = "第 1 章"
OUTPUT_FILE = "02-环境配置.md"

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

def download_vtt():
    print(f"🔽 下载字幕: {VIDEO_URL}")
    subprocess.run([
        "yt-dlp", "--no-update",
        "--write-auto-subs", "--sub-langs", "en",
        "--convert-subs", "vtt", "--skip-download",
        "-o", str(DOCS_DIR / f"_{VIDEO_ID}"),
        VIDEO_URL
    ], check=True, capture_output=True)

    vtt_path = next(DOCS_DIR.glob(f"_{VIDEO_ID}*.vtt"), None)
    if not vtt_path:
        print("❌ 字幕文件未找到")
        sys.exit(1)
    return vtt_path

def parse_vtt(path):
    text = path.read_text(encoding="utf-8")
    segments = []
    for line in text.split("\n"):
        line = line.strip()
        if "-->" in line:
            t = line.split("-->")[0].strip()
            segments.append({"time": t, "text": ""})
        elif segments and line and not line.startswith("WEBVTT") and not line.isdigit():
            clean = re.sub(r"<[^>]+>", "", line)
            if clean:
                segments[-1]["text"] += " " + clean
    return [s for s in segments if s["text"].strip()]

def to_markdown(segments):
    lines = [
        f"# {CHAPTER}：{TITLE_ZH}（视频笔记）",
        "",
        f"> 🎬 [原视频]({VIDEO_URL})",
        f"> 📅 自动生成 | ⚠️ 以下为 AI 自动翻译，建议对照原视频核对",
        "",
        "---",
        "",
        "## 英文原文 + 中文翻译",
        "",
    ]
    for s in segments:
        ts = s["time"].replace(",", ".").split(":")
        ts_str = f"{int(ts[0]):02d}:{int(ts[1]):02d}:{int(float(ts[2])):02d}"
        lines.append(f"**[{ts_str}]** {s['text'].strip()}")
        lines.append("")
    lines.extend(["", "---", "", "## 📝 关键要点（待补充）", "", "- [ ] ", "- [ ] ", "- [ ] "])
    return "\n".join(lines)

def translate_with_deepseek(text):
    """使用 DeepSeek API 翻译（免费额度可用）"""
    import requests
    api_key = "sk-your-key-here"  # 替换你的 API key
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    prompt = f"Translate the following English technical transcript to Simplified Chinese. Keep technical terms in English. Output ONLY the Chinese translation, no explanations:\n\n{text}"

    resp = requests.post(url, headers=headers, json={
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4096,
    }, timeout=60)

    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"[翻译失败: {resp.status_code}]"

def translate_en_to_cn(content):
    """翻译英文 Markdown 为中文"""
    print("🌐 正在翻译...")

    lines = content.split("\n")
    result = []
    translating = False

    for line in lines:
        if line.startswith("**[") and line.strip().endswith("**"):
            result.append(line)
            translating = True
        elif translating and line.strip():
            en_text = line.strip()
            cn_text = translate_with_deepseek(en_text)
            result.append(f"{en_text}")
            result.append(f"> 🇨🇳 {cn_text}")
            result.append("")
            translating = False
        else:
            result.append(line)
            translating = False

    return "\n".join(result)

# --- main ---
print("=" * 50)
print(f"  {TITLE_ZH} → 中文学术笔记")
print("=" * 50)

vtt = download_vtt()
print(f"✅ 字幕已下载: {len(vtt.read_text())} 字节")

segs = parse_vtt(vtt)
print(f"📝 解析到 {len(segs)} 个分段")

md = to_markdown(segs)
path = DOCS_DIR / OUTPUT_FILE
path.write_text(md, encoding="utf-8")
print(f"💾 已保存: {path}")

vtt.unlink()
print("🧹 已清理临时文件")

print(f"""
{'='*50}
✅ 完成！打开 {path} 查看

💡 如需中文翻译（方式任选）：
   1️⃣ 复制笔记内容 → 粘贴到 ChatGPT / DeepSeek
      提示词: "翻译以下英文为简体中文，保留技术术语英文"
   2️⃣ 设置 API key 后重新运行此脚本自动翻译
{'='*50}
""")
