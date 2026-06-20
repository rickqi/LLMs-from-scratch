#!/usr/bin/env python3
"""
DPO 偏好数据准备 — 基于已有 QA 对生成 chosen/rejected 偏好对
"""

import json, random, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

QA_FILE = Path("docs/med_qa_cases.json")
OUTPUT = Path("data/dpo_pairs.json")


def degrade_response(text: str, method: str = "truncate") -> str:
    """生成劣质回答"""
    if method == "truncate":
        # 截断后半部分
        mid = len(text) // 2
        return text[:mid] + "……"
    elif method == "shuffle_sentences":
        # 随机打乱句子
        sentences = text.replace("。", "。|").split("|")
        random.shuffle(sentences)
        return "".join(sentences)
    elif method == "generic":
        # 替换为通用回答
        return "根据临床经验，建议进一步检查后确定治疗方案。"
    return text[:len(text)//3] + "..."


def main():
    with open(QA_FILE, encoding="utf-8") as f:
        qa_data = json.load(f)

    cases = qa_data.get("cases", qa_data) if isinstance(qa_data, dict) else qa_data

    pairs = []
    for item in cases:
        question = item.get("question", "")
        answer = item.get("answer", "")

        if len(question) < 10 or len(answer) < 50:
            continue

        # Chosen: 原始高质量回答
        chosen = answer

        # Rejected: 随机选择3种劣化方式之一
        method = random.choice(["truncate", "shuffle_sentences", "generic"])
        rejected = degrade_response(answer, method)

        pairs.append({
            "prompt": question,
            "chosen": chosen,
            "rejected": rejected,
        })

    # 保存
    OUTPUT.parent.mkdir(exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"Generated {len(pairs)} DPO preference pairs")
    logger.info(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
