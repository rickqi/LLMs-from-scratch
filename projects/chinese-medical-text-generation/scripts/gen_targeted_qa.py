#!/usr/bin/env python3
"""
针对性 QA 生成 — 为 TNM/影像/术后/甲状腺等弱覆盖领域生成 QA 对。
从源文档中提取相关段落，构造 Q&A 对。

用法:
  python scripts/gen_targeted_qa.py          # 生成并合并
  python scripts/gen_targeted_qa.py --dry-run # 仅预览，不生成
"""

import json, re, random, os, hashlib
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_MEDICA = Path("/home/raw/medica")
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_FILE = DOCS_DIR / "med_instruction_chatml.json"

# ── 目标领域配置 ──────────────────────────────────────────────
TARGET_TOPICS = {
    "TNM肿瘤分期": {
        "keywords": ["TNM", "T1", "T2", "T3", "T4", "N0", "N1", "N2", "N3", "M0", "M1",
                     "分期标准", "临床分期", "病理分期", "Stage", "Tis"],
        "questions": [
            "{cancer}的TNM分期标准是什么？请详细说明T、N、M各期的定义。",
            "根据TNM分期，{cancer}的I期、II期、III期、IV期如何划分？",
            "{cancer}的T分期如何根据肿瘤大小和侵犯范围进行判定？",
            "{cancer}的N分期中，N1、N2、N3分别代表什么淋巴结转移情况？",
            "M0和M1在{cancer}的TNM分期中分别代表什么？",
            "在{cancer}的治疗决策中，TNM分期如何影响手术方案的选择？",
            "请对比{cancer}的临床分期(cTNM)和病理分期(pTNM)的差异和临床意义。",
            "{cancer}患者术前影像学评估如何帮助确定cTNM分期？",
            "根据最新版TNM分期，{cancer}中T4期包括哪些情况？",
            "早期{cancer}（T1N0M0）的标准治疗方案是什么？后期（T4N2M1）如何调整？",
        ],
    },
    "影像学对比": {
        "keywords": ["CT", "MRI", "磁共振", "超声", "PET-CT", "增强扫描", "影像学检查",
                     "影像学表现", "影像诊断", "X线", "CT扫描", "MRI扫描"],
        "questions": [
            "在{cancer}的诊断中，CT和MRI各自的优势是什么？",
            "请对比增强CT和MRI在{cancer}分期中的准确性。",
            "超声检查在{cancer}的初步筛查中有何价值？与CT相比有何优劣？",
            "PET-CT在{cancer}的远处转移评估中扮演什么角色？",
            "对于{cancer}患者，选择CT还是MRI作为首选影像学检查的依据是什么？",
            "在{cancer}的术后随访中，CT和MRI的使用频率和指征有何不同？",
            "请对比CT和MRI在检测{cancer}淋巴结转移中的敏感性和特异性。",
            "{cancer}的影像学鉴别诊断中，哪些特征有助于区分良性和恶性病变？",
            "对于{cancer}，CT增强扫描的典型表现有哪些？如何解读？",
            "MRI在{cancer}的软组织侵犯评估中相比CT有哪些优势？",
        ],
    },
    "术后并发症": {
        "keywords": ["术后", "并发症", "手术后", "切除后", "术后管理", "术后护理",
                     "术后观察", "术后并发症", "手术并发症"],
        "questions": [
            "{surgery}术后常见并发症有哪些？如何预防和处理？",
            "{surgery}术后感染的危险因素和预防措施是什么？",
            "{surgery}术后出血的临床表现和处理原则是什么？",
            "{surgery}术后深静脉血栓的预防策略包括哪些？",
            "{surgery}术后疼痛管理的原则和方法有哪些？",
            "{surgery}术后早期下床活动的时机和注意事项是什么？",
            "{surgery}术后吻合口漏的早期识别和处理方案是什么？",
            "{surgery}术后应如何监测生命体征和引流情况？",
            "{surgery}术后常见的神经系统并发症有哪些？",
            "{surgery}术后发热的鉴别诊断和处理流程是什么？",
        ],
    },
    "甲状腺疾病": {
        "keywords": ["甲状腺", "甲状腺癌", "甲状腺切除", "甲状腺功能",
                     "甲状腺结节", "甲亢", "甲状旁腺", "TSH", "T3", "T4"],
        "questions": [
            "甲状腺癌的病理分型及各型的临床特点是什么？",
            "甲状腺切除手术的适应证和禁忌证有哪些？",
            "甲状腺全切术后长期管理包括哪些内容？",
            "甲状腺癌术后TSH抑制治疗的目标和原则是什么？",
            "甲状腺结节的良恶性鉴别诊断流程是什么？",
            "甲状腺癌术后甲状旁腺功能减退的预防和处理？",
            "分化型甲状腺癌术后放射性碘治疗的适应证是什么？",
            "甲状腺癌术后复发的高危因素和随访策略是什么？",
            "甲状腺手术中喉返神经的保护策略有哪些？",
            "甲状腺癌靶向治疗的适应证和常用药物有哪些？",
        ],
    },
}

# ── 癌症类型占位符 ──────────────────────────────────────────────
CANCER_TYPES = [
    "肺癌", "胃癌", "肝癌", "乳腺癌", "结直肠癌", "食管癌", "胰腺癌",
    "宫颈癌", "卵巢癌", "前列腺癌", "膀胱癌", "肾癌", "甲状腺癌",
    "鼻咽癌", "口腔癌", "喉癌", "脑胶质瘤", "骨肉瘤", "黑色素瘤", "淋巴瘤",
]

SURGERY_TYPES = [
    "胃癌根治术", "肺癌切除术", "肝切除术", "乳腺癌根治术",
    "结直肠癌根治术", "甲状腺切除术", "胰腺十二指肠切除术",
    "前列腺癌根治术", "全子宫切除术", "膀胱全切术",
]


def extract_passages(filepath: Path, keywords: list[str], context_chars: int = 500) -> list[str]:
    """从文档中提取包含关键词的段落"""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    passages = []
    for kw in keywords:
        for match in re.finditer(re.escape(kw), text, re.IGNORECASE):
            start = max(0, match.start() - context_chars // 2)
            end = min(len(text), match.end() + context_chars // 2)
            # Extend to paragraph boundaries
            while start > 0 and text[start - 1] != "\n":
                start -= 1
            while end < len(text) and text[end] != "\n":
                end += 1
            passage = text[start:end].strip()
            if len(passage) > 50:
                passages.append(passage)
    return passages


def generate_qa_for_topic(topic_name: str, config: dict, max_qa: int = 50) -> list[dict]:
    """为特定主题生成 QA 对"""
    print(f"\n  📚 {topic_name}: 扫描源文档...")
    keyword_docs = set()
    for kw in config["keywords"]:
        for f in RAW_MEDICA.rglob("*.md"):
            if kw.lower() in f.name.lower():
                keyword_docs.add(f)
            else:
                try:
                    # Quick grep
                    if kw in f.read_text(encoding="utf-8", errors="replace"):
                        keyword_docs.add(f)
                except Exception:
                    pass
        if len(keyword_docs) >= 30:
            break

    print(f"    匹配 {len(keyword_docs)} 篇文档")

    # Extract passages
    all_passages = []
    for doc in list(keyword_docs)[:20]:
        passages = extract_passages(doc, config["keywords"][:5])
        all_passages.extend(passages)

    print(f"    抽取 {len(all_passages)} 个相关段落")

    # Select diverse passages (by hash to avoid duplicates)
    seen = set()
    diverse = []
    for p in sorted(all_passages, key=len, reverse=True):
        h = hashlib.md5(p[:100].encode()).hexdigest()
        if h not in seen and 100 < len(p) < 1500:
            seen.add(h)
            diverse.append(p)
        if len(diverse) >= max_qa * 2:
            break

    print(f"    去重后 {len(diverse)} 个段落")

    # Generate QA pairs
    qa_pairs = []
    used_questions = set()
    random.seed(42)

    for i, passage in enumerate(diverse[:max_qa * 2]):
        template = random.choice(config["questions"])
        cancer = random.choice(CANCER_TYPES)
        surgery = random.choice(SURGERY_TYPES)
        question = template.format(cancer=cancer, surgery=surgery)

        if question in used_questions:
            continue
        used_questions.add(question)

        # Create answer from passage
        answer = passage[:800].strip()
        # Truncate at sentence boundary
        if len(answer) > 700:
            last_period = answer[:700].rfind("。")
            if last_period > 200:
                answer = answer[:last_period + 1].strip()

        qa_pairs.append({
            "question": question,
            "answer": answer,
            "source": topic_name,
        })

        if len(qa_pairs) >= max_qa:
            break

    print(f"    生成 {len(qa_pairs)} 条 QA")
    return qa_pairs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--per-topic", type=int, default=50,
                       help="每个主题生成的 QA 数 (默认 50)")
    args = parser.parse_args()

    if args.dry_run:
        print("🔍 预览模式 (不实际生成)")

    all_new_qa = []
    for topic_name, config in TARGET_TOPICS.items():
        qa = generate_qa_for_topic(topic_name, config, max_qa=args.per_topic)
        all_new_qa.extend(qa)

    print(f"\n{'='*60}")
    print(f"  共生成 {len(all_new_qa)} 条新 QA")
    print(f"{'='*60}")

    if args.dry_run:
        print("  [DRY-RUN] 未写入文件")
        for qa in all_new_qa[:5]:
            print(f"\n  Q: {qa['question'][:80]}...")
            print(f"  A: {qa['answer'][:100]}...")
        return

    # Load existing ChatML data
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)
        existing_data = existing.get("data", [])
        existing_questions = {item.get("question", item.get("instruction", "")) for item in existing_data}
        print(f"\n现有 QA: {len(existing_data)} 条")
    else:
        existing = {"format": "chatml", "data": []}
        existing_data = []
        existing_questions = set()
        print(f"\n无现有数据，创建新文件")

    # Convert new QA to ChatML format and merge
    new_count = 0
    for qa in all_new_qa:
        if qa["question"] not in existing_questions:
            chatml_item = {
                "messages": [
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]},
                ]
            }
            existing_data.append(chatml_item)
            existing_questions.add(qa["question"])
            new_count += 1

    existing["data"] = existing_data
    existing["total"] = len(existing_data)
    print(f"新增: {new_count} 条 (跳过 {len(all_new_qa) - new_count} 条重复)")
    print(f"合并后: {len(existing_data)} 条")

    # Save merged data
    if OUTPUT_FILE.exists():
        backup = OUTPUT_FILE.with_suffix(".json.bak")
        OUTPUT_FILE.rename(backup)
        print(f"备份: {backup}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"保存: {OUTPUT_FILE}")

    # Also update med_qa_cases.json
    cases_file = DOCS_DIR / "med_qa_cases.json"
    if cases_file.exists():
        with open(cases_file) as f:
            cases = json.load(f)
        if isinstance(cases, dict) and "cases" in cases:
            existing_cq = {c["question"] for c in cases["cases"]}
            for qa in all_new_qa:
                if qa["question"] not in existing_cq:
                    cases["cases"].append({
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "source": qa.get("source", "targeted"),
                    })
            cases["metadata"]["updated"] = datetime.now().isoformat()
            with open(cases_file, "w", encoding="utf-8") as f:
                json.dump(cases, f, ensure_ascii=False, indent=2)
            print(f"更新: {cases_file} ({len(cases['cases'])} 条)")


if __name__ == "__main__":
    main()
