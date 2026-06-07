#!/usr/bin/env python3
"""
从已有 .md 字幕文件生成中英文双语 SRT（DeepSeek 翻译）

用法：python YouTube/make_bilingual_srt.py
"""

import re, sys, os, time
from pathlib import Path
from openai import OpenAI

MD_FILES = [
    ("02-环境配置.md",        "02-环境配置"),
    ("03-LLM生命周期.md",     "03-LLM生命周期"),
    ("04-文本数据处理.md",    "04-文本数据处理"),
    ("05-注意力机制.md",      "05-注意力机制"),
    ("06-PyTorch-Buffers.md", "06-PyTorch-Buffers"),
    ("07-GPT模型实现.md",     "07-GPT模型实现"),
    ("08-预训练.md",         "08-预训练"),
    ("09-分类微调.md",       "09-分类微调"),
    ("10-指令微调.md",       "10-指令微调"),
]

SCRIPT_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 80

def parse_md(path):
    text = path.read_text(encoding="utf-8")
    segments = []
    idx = 1
    for m in re.finditer(r"\*\*\[(\d{2}):(\d{2}):(\d{2})\]\*\*\s*(.+)", text):
        h, mi, s, content = m.group(1), m.group(2), m.group(3), m.group(4).strip()
        if not content:
            continue
        ts = f"00:{h}:{mi},{s}00"
        segments.append({
            "index": idx,
            "start": ts,
            "end": ts,
            "en": content,
        })
        idx += 1
    for i in range(len(segments) - 1):
        segments[i]["end"] = segments[i + 1]["start"]
    return segments

SYSTEM_PROMPT = """You are a professional English-to-Chinese translator specializing in AI/ML technical content.
Translate each numbered English line to Simplified Chinese.
Rules:
- Keep technical terms in English: attention, transformer, GPT, LLM, token, embedding, fine-tuning, pre-training, batch, gradient, epoch, PyTorch, tensor, etc.
- Output ONLY the numbered Chinese lines, one per line, matching the input numbering exactly.
- Do NOT add explanations or extra text.
- Example input: 1. attention mechanism\n2. fine-tuning the model
- Example output: 1. 注意力机制\n2. 微调模型"""

client = OpenAI(
    api_key="sk-bea2ca4afd3d4fc0bd2b6c0a63fe6912",
    base_url="https://api.deepseek.com",
)

def translate_batch(en_lines):
    numbered = "\n".join(f"{i+1}. {line}" for i, line in enumerate(en_lines))
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": numbered},
                ],
                temperature=0.1,
                max_tokens=8192,
            )
            text = r.choices[0].message.content.strip()
            zh_lines = {}
            for m in re.finditer(r"^(\d+)\.\s*(.+)$", text, re.MULTILINE):
                zh_lines[int(m.group(1)) - 1] = m.group(2)
            return [zh_lines.get(i, en_lines[i]) for i in range(len(en_lines))]
        except Exception as e:
            print(f"      retry {attempt+1}: {e}")
            time.sleep(5)
    return list(en_lines)

def translate_all(texts):
    total = len(texts)
    results = [""] * total
    for start in range(0, total, BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        zh = translate_batch(batch)
        for i, val in enumerate(zh):
            results[start + i] = val
        done = min(start + BATCH_SIZE, total)
        print(f"    {done}/{total}")
        time.sleep(0.5)
    return results

def write_bilingual_srt(segments, output_path):
    lines = []
    for s in segments:
        lines.append(str(s["index"]))
        lines.append(f"{s['start']} --> {s['end']}")
        lines.append(s["en"])
        lines.append(s["zh"])
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")

success = 0
for md_name, title in MD_FILES:
    print(f"\n{'='*50}\n  {title}\n{'='*50}")
    md_path = SCRIPT_DIR / md_name
    srt_path = SCRIPT_DIR / f"{title}.bilingual.srt"
    if not md_path.exists():
        print(f"  ⚠️ 源文件不存在: {md_name}")
        continue
    if srt_path.exists():
        print(f"  ⏭️ 已存在，跳过")
        success += 1
        continue

    segments = parse_md(md_path)
    print(f"  📝 {len(segments)} 段")

    texts = [s["en"] for s in segments]
    zh_texts = translate_all(texts)
    for i, zh in enumerate(zh_texts):
        segments[i]["zh"] = zh

    write_bilingual_srt(segments, srt_path)
    size_kb = srt_path.stat().st_size // 1024
    print(f"  💾 {srt_path.name} ({size_kb}KB)")
    success += 1

print(f"\n{'='*50}")
print(f"✅ {success}/{len(MD_FILES)}")
print(f"{'='*50}")
