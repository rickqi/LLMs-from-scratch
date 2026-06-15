"""
数据准备脚本: 公司规章制度文本生成
==================================
用法:
  # 真实数据
  python data_prep.py --data_dir "/home/docs/raw/公司规章制度" --output_dir ./data

  # 自动清洗 -> train.txt / val.txt (===SEP=== 分隔)
"""

import re
import os
import sys
import argparse
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def clean_regulation_text(text: str) -> str:
    """清洗公司制度文本: 去除 PDF/OCR 残留、HTML、多余空行"""
    # HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 图片标记
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # 页码 / OCR bbox
    text = re.sub(r"page=\d+[^)]*\)", "", text)
    # Markdown 标题符号 (保留文本, 去掉 #)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # 水印 / 版本控制信息
    text = re.sub(r"^>\s*.*$", "", text, flags=re.MULTILINE)
    # 多余空行
    text = re.sub(r"\n{4,}", "\n\n", text)
    # 行首尾空格
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return "\n".join(lines)


def load_raw_data(data_dir: str, min_length: int = 100) -> list[str]:
    """递归加载目录下所有 .md 文件, 返回清洗后的文本列表"""
    data_dir = Path(data_dir)
    # 排除 index 目录和 .json 文件
    files = sorted(
        f for f in data_dir.rglob("*.md")
        if "index" not in str(f) and not str(f).endswith(".json")
    )
    logger.info(f"发现 {len(files)} 个 .md 文件")

    texts = []
    skipped = 0
    for fpath in files:
        try:
            raw = fpath.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_regulation_text(raw)
            if len(cleaned) >= min_length:
                texts.append(cleaned)
            else:
                skipped += 1
        except Exception as e:
            logger.warning(f"读取 {fpath.name} 失败: {e}")
            skipped += 1

    logger.info(f"有效文本: {len(texts)} 篇 (跳过 {skipped} 篇过短/错误)")
    logger.info(f"总字符数: {sum(len(t) for t in texts):,}")
    return texts


def split_train_val(texts: list[str], val_ratio: float = 0.05, seed: int = 42):
    rng = random.Random(seed)
    shuffled = list(texts)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split], shuffled[split:]


def save_data(texts: list[str], output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n\n===SEP===\n\n".join(texts)
    output_path.write_text(content, encoding="utf-8")
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"保存 {output_path}: {len(texts)} 篇, {size_mb:.1f} MB")


def show_stats(texts: list[str], label: str):
    lengths = [len(t) for t in texts]
    logger.info(f"\n{'='*50}")
    logger.info(f"  {label}")
    logger.info(f"  总文本数:   {len(texts)}")
    logger.info(f"  总字符数:   {sum(lengths):,}")
    logger.info(f"  平均字符数: {sum(lengths)//max(len(texts),1):,}")
    logger.info(f"  最短/最长:  {min(lengths)} / {max(lengths)}")
    logger.info(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="公司规章制度数据准备")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="原始 .md 文件目录")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="输出目录")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="验证集比例")
    parser.add_argument("--min_length", type=int, default=100,
                        help="最短有效文本长度 (字符)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    texts = load_raw_data(args.data_dir, args.min_length)

    if not texts:
        logger.error("未找到有效文本!")
        sys.exit(1)

    show_stats(texts, "原始数据")

    train_texts, val_texts = split_train_val(texts, args.val_ratio, args.seed)
    show_stats(train_texts, "训练集")
    show_stats(val_texts, "验证集")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_data(train_texts, output_dir / "train.txt")
    save_data(val_texts, output_dir / "val.txt")

    logger.info(f"\n数据准备完成! 下一步:")
    logger.info(f"  python train_qwen_lora.py --data_dir {output_dir} --output_dir ./output")


if __name__ == "__main__":
    main()
