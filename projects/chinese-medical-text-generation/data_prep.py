"""
数据准备脚本: 中文医学诊疗指南文本生成
========================================
用法:
  python data_prep.py --data_dir ./raw_data --output_dir ./data

数据格式: 接受 .md 文件目录, 每个文件是一份医学指南
"""

import re
import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def clean_medical_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'ISBN[^\n]*', '', text)
    text = re.sub(r'page=\d+[^)]*\)', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\n{4,}', '\n\n', text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return '\n'.join(lines)


def load_raw_data(data_dir: str) -> list[str]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.rglob("*.md"))
    logger.info(f"发现 {len(files)} 个 .md 文件")

    texts = []
    for fpath in files:
        raw = fpath.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_medical_text(raw)
        if len(cleaned) > 50:
            texts.append(cleaned)
    logger.info(f"清洗后有效文本: {len(texts)} 篇, 总字符数: {sum(len(t) for t in texts):,}")
    return texts


def split_train_val(texts: list[str], val_ratio: float = 0.05):
    random.shuffle(texts)
    split = int(len(texts) * (1 - val_ratio))
    return texts[:split], texts[split:]


def save_data(texts: list[str], output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n\n===SEP===\n\n".join(texts),
        encoding="utf-8"
    )
    logger.info(f"保存到 {output_path}, 共 {len(texts)} 篇, {output_path.stat().st_size:,} 字节")


def main():
    parser = argparse.ArgumentParser(description="准备中文医学文本数据")
    parser.add_argument("--data_dir", type=str, required=True, help="原始 .md 文件目录")
    parser.add_argument("--output_dir", type=str, default="./data", help="输出目录")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例")
    args = parser.parse_args()

    texts = load_raw_data(args.data_dir)
    if not texts:
        logger.error("未找到有效文本, 请检查 data_dir")
        return

    train_texts, val_texts = split_train_val(texts, args.val_ratio)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data(train_texts, output_dir / "train.txt")
    save_data(val_texts, output_dir / "val.txt")

    # 输出统计信息
    logger.info(f"\n{'='*50}")
    logger.info(f"  总文本数:   {len(texts)}")
    logger.info(f"  训练集:     {len(train_texts)} 篇 ({sum(len(t) for t in train_texts):,} 字符)")
    logger.info(f"  验证集:     {len(val_texts)} 篇 ({sum(len(t) for t in val_texts):,} 字符)")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    import random
    main()
