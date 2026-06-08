#!/usr/bin/env python3
"""Preprocess markdown files for HonKit ebook build.
Strips external image references to avoid network timeouts.
"""

import os
import re
import shutil
import sys

SRC = "/home/LLMs-from-scratch"
DST = "/home/LLMs-from-scratch/ebook-build"

STRIP_MARKDOWN_IMAGES = re.compile(r'!\[.*?\]\(https?://[^)]+\)')
STRIP_LINKED_IMAGES = re.compile(r'\[!\[.*?\]\(https?://[^)]+\)\]\([^)]+\)')
STRIP_HTML_IMG = re.compile(r'<img\s+[^>]*src=["\']https?://[^"\']+["\'][^>]*>')
STRIP_HTML_A_IMG = re.compile(r'<a\s+[^>]*href=["\'][^"\']+["\'][^>]*>\s*<img[^>]*>\s*</a>', re.IGNORECASE)

FILES = [
    "README_CN.md",
    "troubleshooting_CN.md",
    "ANALYSIS_CN.md",
    "setup/README_CN.md",
    "setup/01_optional-python-setup-preferences/README_CN.md",
    "setup/02_installing-python-libraries/README_CN.md",
    "setup/03_optional-docker-environment/README_CN.md",
    "setup/04_optional-aws-sagemaker-notebook/README_CN.md",
    "docs/01-学习路线与建议.md",
    "docs/02-环境配置.md",
    "docs/03-LLM生命周期与第2章预习.md",
    "docs/04-注意力机制.md",
    "docs/05-GPT模型实现.md",
    "docs/06-预训练.md",
    "docs/07-分类微调.md",
    "docs/08-指令微调.md",
    "docs/09-PyTorch入门.md",
    "docs/10-LoRA参数高效微调.md",
    "docs/youtube/02-环境配置.md",
    "docs/youtube/03-LLM生命周期.md",
    "docs/youtube/04-文本数据处理.md",
    "docs/youtube/05-注意力机制.md",
    "docs/youtube/06-PyTorch-Buffers.md",
    "docs/youtube/07-GPT模型实现.md",
    "docs/youtube/08-预训练.md",
    "docs/youtube/09-分类微调.md",
    "docs/youtube/10-指令微调.md",
    "ch01/README_CN.md",
    "ch02/README_CN.md",
    "ch02/01_main-chapter-code/README_CN.md",
    "ch02/02_bonus_bytepair-encoder/README_CN.md",
    "ch02/03_bonus_embedding-vs-matmul/README_CN.md",
    "ch02/04_bonus_dataloader-intuition/README_CN.md",
    "ch02/05_bpe-from-scratch/README_CN.md",
    "ch03/README_CN.md",
    "ch03/01_main-chapter-code/README_CN.md",
    "ch03/02_bonus_efficient-multihead-attention/README_CN.md",
    "ch03/03_understanding-buffers/README_CN.md",
    "ch04/README_CN.md",
    "ch04/01_main-chapter-code/README_CN.md",
    "ch04/02_performance-analysis/README_CN.md",
    "ch04/03_kv-cache/README_CN.md",
    "ch04/04_gqa/README_CN.md",
    "ch04/05_mla/README_CN.md",
    "ch04/06_swa/README_CN.md",
    "ch04/07_moe/README_CN.md",
    "ch04/08_deltanet/README_CN.md",
    "ch04/09_dsa/README_CN.md",
    "ch04/10_kv-sharing/README_CN.md",
    "ch05/README_CN.md",
    "ch05/01_main-chapter-code/README_CN.md",
    "ch05/02_alternative_weight_loading/README_CN.md",
    "ch05/03_bonus_pretraining_on_gutenberg/README_CN.md",
    "ch05/04_learning_rate_schedulers/README_CN.md",
    "ch05/05_bonus_hparam_tuning/README_CN.md",
    "ch05/06_user_interface/README_CN.md",
    "ch05/07_gpt_to_llama/README_CN.md",
    "ch05/08_memory_efficient_weight_loading/README_CN.md",
    "ch05/09_extending-tokenizers/README_CN.md",
    "ch05/10_llm-training-speed/README_CN.md",
    "ch05/11_qwen3/README_CN.md",
    "ch05/11_qwen3/qwen3-chat-interface/README_CN.md",
    "ch05/12_gemma3/README_CN.md",
    "ch05/13_olmo3/README_CN.md",
    "ch05/14_ch05_with_other_llms/README_CN.md",
    "ch05/15_tiny-aya/README_CN.md",
    "ch05/16_qwen3.5/README_CN.md",
    "ch05/17_gemma4/README_CN.md",
    "ch05/18_muon/README_CN.md",
    "ch06/README_CN.md",
    "ch06/01_main-chapter-code/README_CN.md",
    "ch06/02_bonus_additional-experiments/README_CN.md",
    "ch06/03_bonus_imdb-classification/README_CN.md",
    "ch06/04_user_interface/README_CN.md",
    "ch07/README_CN.md",
    "ch07/01_main-chapter-code/README_CN.md",
    "ch07/02_dataset-utilities/README_CN.md",
    "ch07/03_model-evaluation/README_CN.md",
    "ch07/04_preference-tuning-with-dpo/README_CN.md",
    "ch07/05_dataset-generation/README_CN.md",
    "ch07/06_user_interface/README_CN.md",
    "appendix-A/README_CN.md",
    "appendix-A/01_main-chapter-code/README_CN.md",
    "appendix-A/02_setup-recommendations/README_CN.md",
    "appendix-B/README_CN.md",
    "appendix-C/README_CN.md",
    "appendix-D/README_CN.md",
    "appendix-E/README_CN.md",
    "pkg/llms_from_scratch/README_CN.md",
]


def preprocess(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    content = STRIP_LINKED_IMAGES.sub("", content)
    content = STRIP_HTML_A_IMG.sub("", content)
    content = STRIP_HTML_IMG.sub("", content)
    content = STRIP_MARKDOWN_IMAGES.sub("", content)
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    return content


def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    os.makedirs(DST, exist_ok=True)

    for relpath in FILES:
        src_path = os.path.join(SRC, relpath)
        dst_path = os.path.join(DST, relpath)
        if not os.path.exists(src_path):
            print(f"  SKIP (not found): {relpath}")
            continue
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        content = preprocess(src_path)
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  OK: {relpath}")

    for extra in ["SUMMARY.md", "book.json"]:
        src_extra = os.path.join(SRC, extra)
        if os.path.exists(src_extra):
            shutil.copy2(src_extra, os.path.join(DST, extra))
            print(f"  COPY: {extra}")

    styles_dir = os.path.join(DST, "styles")
    os.makedirs(styles_dir, exist_ok=True)
    for css in os.listdir(os.path.join(SRC, "styles")):
        if css.endswith(".css"):
            shutil.copy2(os.path.join(SRC, "styles", css), os.path.join(styles_dir, css))
            print(f"  COPY: styles/{css}")

    print(f"\nPreprocessed {len(FILES)} files into {DST}")


if __name__ == "__main__":
    main()
