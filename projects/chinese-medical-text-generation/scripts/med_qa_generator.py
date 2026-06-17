#!/usr/bin/env python3
"""
医学指令微调数据集生成器
========================
基于 doc-search qa_benchmark.py 改造，从中文医学文档自动生成高质量 QA 对。

用法:
  python scripts/med_qa_generator.py generate     # 生成 QA 对
  python scripts/med_qa_generator.py stats         # 查看统计
  python scripts/med_qa_generator.py export        # 导出为指令微调格式
  python scripts/med_qa_generator.py all           # 一键全流程

环境变量:
  DEEPSEEK_API_KEY   DeepSeek API Key (默认从 doc-search .env 读取)
  DEEPSEEK_BASE_URL  DeepSeek API URL (默认 https://api.deepseek.com)
"""

import json
import os
import re
import sys
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# ── 项目路径 ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# 数据源
RAW_MEDICA = Path("/home/raw/medica")

# 输出文件
QA_FILE = DOCS_DIR / "med_qa_cases.json"
EXPORT_CHATML_FILE = DOCS_DIR / "med_instruction_chatml.json"
EXPORT_ALPACA_FILE = DOCS_DIR / "med_instruction_alpaca.json"
REPORT_FILE = DOCS_DIR / "med_qa_report.md"

# ═══════════════════════════════════════════════════════════════
# 三档数据集配置
# ═══════════════════════════════════════════════════════════════

# 各来源的文档总量和可用提示词
CATEGORY_CONFIG = {
    "L1_卫健委官方规范": {
        "domain_label": "肿瘤诊疗指南",
        "total_docs": 170,
        "hints": [
            "诊断标准", "治疗方案", "临床表现", "预后判断",
            "化疗方案", "放疗", "靶向治疗", "免疫治疗",
            "TNM分期", "手术指征", "随访", "复发转移",
        ],
    },
    "L2_卫生行业标准_WS": {
        "domain_label": "临床检验与感染控制标准",
        "total_docs": 58,
        "hints": [
            "标本采集", "参考区间", "质量控制", "性能验证",
            "消毒隔离", "手卫生", "医院感染", "安全注射",
            "操作规程", "生物安全", "检验方法", "室间质评",
        ],
    },
    "L3_中华医学会临床诊疗指南丛书": {
        "domain_label": "临床诊疗指南与专家共识",
        "total_docs": 140,
        "hints": [
            "鉴别诊断", "药物治疗", "手术方案", "并发症",
            "专家共识", "诊疗流程", "影像学检查", "病理诊断",
            "分级分期", "适应证", "禁忌证", "疗效评估",
        ],
    },
    "L4_中华医学会临床技术操作规范": {
        "domain_label": "临床技术操作规范",
        "total_docs": 44,
        "hints": [
            "操作步骤", "无菌操作", "术前准备", "术后处理",
            "并发症预防", "器械使用", "麻醉", "护理要点",
        ],
    },
}

# 三档数据集：每档定义各来源的抽样文档数
TIER_CONFIGS = {
    "small": {
        "description": "小数据集 — 快速实验，~2h 训练",
        "estimated_pairs": "300-350",
        "estimated_train_time": "~1.5h (3 epochs)",
        "use_case": "验证脚本正确性 / 快速迭代 prompt / 超参调试",
        "samples": {
            "L1_卫健委官方规范": 18,
            "L2_卫生行业标准_WS": 8,
            "L3_中华医学会临床诊疗指南丛书": 18,
            "L4_中华医学会临床技术操作规范": 10,
        },
    },
    "medium": {
        "description": "中等数据集 — 质量与覆盖的平衡点",
        "estimated_pairs": "600-700",
        "estimated_train_time": "~3h (3 epochs)",
        "use_case": "正式指令微调训练 / 评估指令跟随能力",
        "samples": {
            "L1_卫健委官方规范": 35,
            "L2_卫生行业标准_WS": 5,   # 近枯竭，只补充少量
            "L3_中华医学会临床诊疗指南丛书": 30,
            "L4_中华医学会临床技术操作规范": 15,
        },
    },
    "large": {
        "description": "大数据集 — 最大覆盖，~5h 训练",
        "estimated_pairs": "1000-1200",
        "estimated_train_time": "~5h (3 epochs)",
        "use_case": "最终模型 / 追求最佳指令跟随效果",
        "samples": {
            "L1_卫健委官方规范": 50,
            "L2_卫生行业标准_WS": 5,
            "L3_中华医学会临床诊疗指南丛书": 70,
            "L4_中华医学会临床技术操作规范": 40,
        },
    },
}

# 默认使用 small 档
DEFAULT_TIER = "small"

def get_tier_config(tier_name: str) -> dict:
    """获取指定档位的配置"""
    if tier_name not in TIER_CONFIGS:
        print(f"⚠ 未知档位 '{tier_name}'，使用默认档 'small'")
        print(f"  可用档位: {', '.join(TIER_CONFIGS.keys())}")
        tier_name = DEFAULT_TIER
    return TIER_CONFIGS[tier_name]

# 兼容旧代码：根据 tier 构建 DOMAIN_SAMPLES
def build_domain_samples(tier_name: str) -> dict:
    """从 tier 配置构建 DOMAIN_SAMPLES 格式"""
    tier = get_tier_config(tier_name)
    result = {}
    for dir_name, cfg in CATEGORY_CONFIG.items():
        sample_count = tier["samples"].get(dir_name, 5)
        result[dir_name] = {
            "sample_count": sample_count,
            "domain_label": cfg["domain_label"],
            "hints": cfg["hints"],
        }
    return result

# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class MedQACase:
    """单条医学 QA 案例"""
    id: str
    domain: str
    source_dir: str
    question: str
    answer: str
    source_text: str          # 答案依据的原文段落
    source_file: str          # 来源文档相对路径
    difficulty: str           # easy / medium / hard
    query_type: str           # factual / procedural / diagnostic / comparative
    keywords: List[str] = field(default_factory=list)

# ═══════════════════════════════════════════════════════════════
# LLM 客户端 (OpenAI-compatible / DeepSeek)
# ═══════════════════════════════════════════════════════════════

def _load_api_config() -> dict:
    """加载 API 配置，优先环境变量，其次 doc-search .env"""
    config = {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "model": os.environ.get("LLM_MODEL", "deepseek-chat"),
    }

    # 从 doc-search .env 读取
    doc_env = Path("/mnt/d/docs/doc-search/.env")
    if doc_env.exists() and not config["api_key"]:
        for line in doc_env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                config["api_key"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("DEEPSEEK_BASE_URL="):
                config["base_url"] = line.split("=", 1)[1].strip().strip('"').strip("'")

    return config


def create_llm_client():
    """创建 OpenAI 兼容客户端 (DeepSeek)"""
    from openai import OpenAI

    cfg = _load_api_config()
    if not cfg["api_key"]:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY。请设置环境变量或在 /mnt/d/docs/doc-search/.env 中配置"
        )
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ═══════════════════════════════════════════════════════════════
# 文档选择与内容提取
# ═══════════════════════════════════════════════════════════════

def select_documents(source_dir: Path, sample_count: int) -> List[Path]:
    """选择适合生成 QA 的文档 (3KB-80KB, 偏向中等大小)"""
    candidates = []
    for md in source_dir.rglob("*.md"):
        if md.name.startswith("_") or md.name == "index.md":
            continue
        if md.name.endswith(".json"):
            continue
        size = md.stat().st_size
        if 1000 < size < 120000:
            candidates.append((md, size))
    candidates.sort(key=lambda x: x[1], reverse=True)

    if len(candidates) <= sample_count:
        return [c[0] for c in candidates]

    # 中等大小优先 (10KB-80KB 最适)
    mid = [c for c in candidates if 10000 < c[1] < 80000]
    if len(mid) >= sample_count:
        return [c[0] for c in mid[:sample_count]]
    return [c[0] for c in candidates[:sample_count]]


def extract_content_sections(md_path: Path, max_chars: int = 4000) -> str:
    """提取文档核心正文（跳过封面/目录/版本表等噪音）"""
    text = md_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # 找正文开始
    content_start = 0
    for i, line in enumerate(lines):
        line_s = line.strip()
        # 检测第一章/第一条/第一节
        if re.match(r"(#+\s*)?第[一二三四五六七八九十\d]+[章节条篇]", line_s):
            content_start = i
            break
        # 检测 "1 范围" / "一、概述" 等
        if re.match(r"(#+\s*)?[一二三四五六七八九十\d]+[\.\s、)]", line_s) and i > 10:
            content_start = i
            break
        # 检测 "一、概述" / "1. 范围" 等
        if re.match(r"(#+\s*)?(一|二|三|四|五|六|七|八|九|十)[、．.]", line_s) and i > 5:
            content_start = i
            break

    content_lines = lines[content_start:]
    content = "\n".join(content_lines).strip()

    # 跳过仅含 HTML 注释的行
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

    if len(content) > max_chars:
        content = content[:max_chars]

    return content


# ═══════════════════════════════════════════════════════════════
# QA 生成 Prompt
# ═══════════════════════════════════════════════════════════════

MEDICAL_QA_PROMPT = """你是一位临床医学教育专家和医学考试命题专家。请根据以下医学文档内容，生成 3-5 个高质量的问答对，用于训练医学 AI 助手。

## 核心要求

1. **问题多样化**：
   - factual (事实查询): "XX疾病的诊断标准是什么？"
   - procedural (流程步骤): "XX手术的术前准备包括哪些步骤？"
   - diagnostic (诊断推理): "患者出现XX症状，应考虑哪些鉴别诊断？"
   - comparative (对比分析): "XX和YY在治疗方案上有何异同？"

2. **难度分布**：
   - easy (1-2个): 答案可直接从文档原文找到
   - medium (2个): 需要理解上下文并组织信息
   - hard (1个): 需要综合多段信息进行推理

3. **质量要求**：
   - 答案必须严格基于文档原文，不得编造任何医学事实
   - 每个QA对必须包含 source_text（答案所依据的原文片段，至少80字）
   - 问题应模拟临床医生或医学生实际会提出的问题
   - 使用专业准确的医学术语

4. **领域聚焦**：{hints_str}

## 文档信息
- 来源: {source_name}
- 领域: {domain_label}

## 文档内容
{content}

## 输出格式
严格按以下 JSON 格式输出，不要输出其他内容：
```json
[
  {{
    "question": "具体问题（以问号结尾）",
    "answer": "基于文档的完整答案",
    "source_text": "答案所依据的原文片段（至少80字）",
    "difficulty": "easy|medium|hard",
    "query_type": "factual|procedural|diagnostic|comparative",
    "keywords": ["关键词1", "关键词2", "关键词3"]
  }}
]
```"""


def generate_qa_for_document(
    client,
    model: str,
    doc_path: Path,
    source_dir: str,
    domain_label: str,
    hints: List[str],
    case_id_prefix: str,
) -> List[MedQACase]:
    """用 LLM 从单个文档生成 QA 案例"""
    content = extract_content_sections(doc_path)
    if len(content) < 300:
        return []

    source_name = doc_path.stem[:80]
    hints_str = "、".join(hints[:10]) if hints else "临床医学综合"
    domain_hint = f"{source_dir} — {domain_label}"

    prompt = MEDICAL_QA_PROMPT.format(
        hints_str=hints_str,
        source_name=source_name,
        domain_label=domain_hint,
        content=content,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=3000,
        )
        raw = response.choices[0].message.content.strip()

        # 提取 JSON
        json_match = re.search(r"```json\s*(.*?)```", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1)
        elif not raw.startswith("["):
            json_match2 = re.search(r"\[.*\]", raw, re.DOTALL)
            if json_match2:
                raw = json_match2.group(0)
            else:
                print(f"    ⚠ 无法解析 JSON from {source_name}: {raw[:100]}...")
                return []

        items = json.loads(raw)
        cases = []
        for i, item in enumerate(items):
            case = MedQACase(
                id=f"{case_id_prefix}_{i+1:02d}",
                domain=domain_label,
                source_dir=source_dir,
                question=item["question"],
                answer=item["answer"],
                source_text=item.get("source_text", ""),
                source_file=str(doc_path.relative_to(RAW_MEDICA)),
                difficulty=item.get("difficulty", "medium"),
                query_type=item.get("query_type", "factual"),
                keywords=item.get("keywords", []),
            )
            cases.append(case)
        return cases

    except json.JSONDecodeError as e:
        print(f"    ⚠ JSON 解析错误 from {source_name}: {e}")
        return []
    except Exception as e:
        print(f"    ⚠ 生成错误 from {source_name}: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# 增量保存
# ═══════════════════════════════════════════════════════════════

def _load_existing_cases() -> tuple[list[dict], set[str]]:
    if not QA_FILE.exists():
        return [], set()
    data = json.loads(QA_FILE.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    stems = {c["source_file"].split("/")[-1].replace(".md", "") for c in cases}
    return cases, stems


def _save_cases(cases: list[dict], model: str):
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_cases": len(cases),
            "model": model,
            "domains": list(set(c["domain"] for c in cases)),
        },
        "cases": cases,
    }
    QA_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════
# 命令: generate
# ═══════════════════════════════════════════════════════════════

def cmd_generate(tier: str = "small"):
    """生成医学指令微调 QA 数据"""
    tier_cfg = get_tier_config(tier)
    domain_samples = build_domain_samples(tier)

    print("=" * 60)
    print(f"  医学指令微调 QA 数据生成 [{tier.upper()} 档]")
    print(f"  {tier_cfg['description']}")
    print(f"  预计产出: {tier_cfg['estimated_pairs']} 对, 训练: {tier_cfg['estimated_train_time']}")
    print("=" * 60)

    if not RAW_MEDICA.exists():
        print(f"❌ 数据源不存在: {RAW_MEDICA}")
        sys.exit(1)

    client, model = create_llm_client()
    print(f"📡 LLM: {model}")

    existing_cases, processed_stems = _load_existing_cases()
    print(f"📊 已有 {len(existing_cases)} 案例, {len(processed_stems)} 文档已处理")

    all_cases = list(existing_cases)
    case_counter = len(existing_cases)
    total_generated = 0

    for source_dir_name, cfg in domain_samples.items():
        source_dir = RAW_MEDICA / source_dir_name
        if not source_dir.exists():
            print(f"\n⚠ {source_dir_name} 不存在, 跳过")
            continue

        sample_count = cfg["sample_count"]
        domain_label = cfg["domain_label"]
        hints = cfg["hints"]

        print(f"\n📂 {source_dir_name} ({domain_label})")
        docs = select_documents(source_dir, sample_count)
        print(f"   选中 {len(docs)} 个文档 (共 {sample_count} 配额)")

        for doc in docs:
            doc_stem = doc.stem[:80]
            if doc_stem in processed_stems:
                continue

            case_counter += 1
            prefix = f"MED{case_counter:04d}"
            short_name = doc_stem[:50]
            print(f"   生成 {short_name}...", end=" ", flush=True)

            cases = generate_qa_for_document(
                client, model, doc, source_dir_name, domain_label, hints, prefix
            )
            print(f"{len(cases)} cases")
            total_generated += len(cases)

            if cases:
                all_cases.extend([asdict(c) for c in cases])
                _save_cases(all_cases, model)

    print(f"\n✅ 本轮生成 {total_generated} 条 → 总计 {len(all_cases)} 条")
    print(f"   → {QA_FILE}")

    # 打印统计
    _print_stats(all_cases)


def _print_stats(cases: list[dict]):
    by_domain = {}
    by_diff = {}
    by_type = {}
    for c in cases:
        by_domain[c["domain"]] = by_domain.get(c["domain"], 0) + 1
        by_diff[c["difficulty"]] = by_diff.get(c["difficulty"], 0) + 1
        by_type[c["query_type"]] = by_type.get(c["query_type"], 0) + 1

    print(f"\n📊 分布统计:")
    print(f"   领域: {json.dumps(by_domain, ensure_ascii=False)}")
    print(f"   难度: {json.dumps(by_diff, ensure_ascii=False)}")
    print(f"   类型: {json.dumps(by_type, ensure_ascii=False)}")


# ═══════════════════════════════════════════════════════════════
# 命令: export — 导出为指令微调格式
# ═══════════════════════════════════════════════════════════════

def cmd_export(fmt: str = "chatml"):
    """导出为指令微调格式 (ChatML / Alpaca)"""
    if not QA_FILE.exists():
        print(f"❌ QA 文件不存在: {QA_FILE}")
        print("   请先运行: python scripts/med_qa_generator.py generate")
        sys.exit(1)

    data = json.loads(QA_FILE.read_text(encoding="utf-8"))
    cases = data["cases"]
    print(f"📊 加载 {len(cases)} 条 QA → 导出为 {fmt} 格式")

    SYSTEM_PROMPT = (
        "你是一位中文医学助手，基于权威诊疗指南、行业标准和临床操作规范提供专业的医学信息。"
        "你的回答应当准确、结构化，并引用具体的医学标准或指南来源。"
    )

    if fmt == "chatml":
        export_file = EXPORT_CHATML_FILE
        instructions = []
        for c in cases:
            instructions.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": c["question"]},
                    {"role": "assistant", "content": c["answer"]},
                ],
                "metadata": {
                    "source_file": c["source_file"],
                    "difficulty": c["difficulty"],
                    "query_type": c["query_type"],
                    "domain": c["domain"],
                },
            })
    elif fmt == "alpaca":
        export_file = EXPORT_ALPACA_FILE
        instructions = []
        for c in cases:
            instructions.append({
                "instruction": c["question"],
                "input": "",
                "output": c["answer"],
                "metadata": {
                    "source_file": c["source_file"],
                    "difficulty": c["difficulty"],
                    "query_type": c["query_type"],
                    "domain": c["domain"],
                },
            })

    output = {
        "format": fmt,
        "system_prompt": SYSTEM_PROMPT,
        "exported_at": datetime.now().isoformat(),
        "total": len(instructions),
        "data": instructions,
    }

    export_file.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ 导出 {len(instructions)} 条 ({fmt}) → {export_file}")


# ═══════════════════════════════════════════════════════════════
# 命令: stats — 数据质量分析
# ═══════════════════════════════════════════════════════════════

def cmd_stats():
    """数据分析与质量报告"""
    if not QA_FILE.exists():
        print(f"❌ QA 文件不存在: {QA_FILE}")
        sys.exit(1)

    data = json.loads(QA_FILE.read_text(encoding="utf-8"))
    cases = data["cases"]
    total = len(cases)

    # 统计
    by_domain = {}
    by_diff = {}
    by_type = {}
    by_source_dir = {}
    avg_q_len = 0
    avg_a_len = 0
    avg_src_len = 0

    for c in cases:
        by_domain[c["domain"]] = by_domain.get(c["domain"], 0) + 1
        by_diff[c["difficulty"]] = by_diff.get(c["difficulty"], 0) + 1
        by_type[c["query_type"]] = by_type.get(c["query_type"], 0) + 1
        by_source_dir[c["source_dir"]] = by_source_dir.get(c["source_dir"], 0) + 1
        avg_q_len += len(c["question"])
        avg_a_len += len(c["answer"])
        avg_src_len += len(c.get("source_text", ""))

    avg_q_len //= max(total, 1)
    avg_a_len //= max(total, 1)
    avg_src_len //= max(total, 1)

    # 质量检查
    quality_issues = []
    for c in cases:
        q = c["question"]
        a = c["answer"]
        if not q.endswith("？") and not q.endswith("?"):
            quality_issues.append(f"{c['id']}: 问题未以问号结尾")
        if len(a) < 50:
            quality_issues.append(f"{c['id']}: 答案过短 ({len(a)} 字)")
        if len(c.get("source_text", "")) < 30:
            quality_issues.append(f"{c['id']}: 原文引用过短")
        if len(c.get("keywords", [])) == 0:
            quality_issues.append(f"{c['id']}: 缺少关键词")

    # 输出报告
    report = f"""# 医学指令微调数据 — 质量分析报告

> 生成时间: {data['metadata']['generated_at']}
> 模型: {data['metadata'].get('model', 'N/A')}

## 总览

| 指标 | 值 |
|------|-----|
| 总 QA 数 | {total} |
| 领域覆盖 | {len(by_domain)} 个 |
| 文档来源数 | {len(by_source_dir)} 个目录 |
| 质量问题 | {len(quality_issues)} 条 |

## 领域分布

| 领域 | 数量 | 占比 |
|------|------|------|
"""
    for domain, count in sorted(by_domain.items(), key=lambda x: -x[1]):
        pct = count / max(total, 1) * 100
        report += f"| {domain} | {count} | {pct:.1f}% |\n"

    report += f"""
## 难度分布

| 难度 | 数量 | 占比 |
|------|------|------|
"""
    for diff in ["easy", "medium", "hard"]:
        count = by_diff.get(diff, 0)
        report += f"| {diff} | {count} | {count/max(total,1)*100:.1f}% |\n"

    report += f"""
## 问题类型分布

| 类型 | 数量 | 占比 |
|------|------|------|
"""
    for qtype, count in sorted(by_type.items(), key=lambda x: -x[1]):
        report += f"| {qtype} | {count} | {count/max(total,1)*100:.1f}% |\n"

    report += f"""
## 平均长度

| 字段 | 平均字符数 |
|------|----------|
| 问题 | {avg_q_len} |
| 答案 | {avg_a_len} |
| 原文引用 | {avg_src_len} |

## 来源目录分布

| 目录 | 数量 |
|------|------|
"""
    for sdir, count in sorted(by_source_dir.items()):
        report += f"| {sdir} | {count} |\n"

    if quality_issues:
        report += f"""
## 质量问题

"""
        for issue in quality_issues[:20]:
            report += f"- {issue}\n"
        if len(quality_issues) > 20:
            report += f"- ... 共 {len(quality_issues)} 条\n"

    # 样本展示
    report += """
## 样本展示

"""
    for c in cases[:5]:
        report += f"""### {c['id']} [{c['difficulty']}] [{c['query_type']}]
**Q**: {c['question']}

**A**: {c['answer'][:300]}{'...' if len(c['answer']) > 300 else ''}

**来源**: {c.get('source_text', '')[:200]}

**文件**: {c['source_file']}

---
"""

    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"✅ 质量报告 → {REPORT_FILE}")
    print(f"\n总览: {total} 条, {len(by_domain)} 领域, {len(quality_issues)} 质量问题")
    print(f"难度: easy={by_diff.get('easy',0)} medium={by_diff.get('medium',0)} hard={by_diff.get('hard',0)}")
    print(f"类型: {json.dumps(by_type, ensure_ascii=False)}")


# ═══════════════════════════════════════════════════════════════
# 命令: sample — 预览文档内容
# ═══════════════════════════════════════════════════════════════

def cmd_sample(source_dir_name: str = "L1_卫健委官方规范"):
    """预览选定文档内容，用于调试 prompt"""
    source_dir = RAW_MEDICA / source_dir_name
    if not source_dir.exists():
        print(f"❌ 目录不存在: {source_dir}")
        sys.exit(1)

    docs = select_documents(source_dir, 3)
    for doc in docs:
        print(f"\n{'='*60}")
        print(f"文件: {doc.relative_to(RAW_MEDICA)}")
        print(f"大小: {doc.stat().st_size:,} bytes")
        print(f"{'='*60}")
        content = extract_content_sections(doc, 1500)
        print(content[:1500])
        print("...")

# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="医学指令微调数据集生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/med_qa_generator.py generate          # 生成 QA 对
  python scripts/med_qa_generator.py stats             # 查看统计
  python scripts/med_qa_generator.py export            # 导出 ChatML 格式
  python scripts/med_qa_generator.py export --fmt alpaca  # 导出 Alpaca 格式
  python scripts/med_qa_generator.py sample L2_卫生行业标准_WS  # 预览文档
  python scripts/med_qa_generator.py all               # 全流程
        """,
    )
    sub = parser.add_subparsers(dest="command")

    gen_p = sub.add_parser("generate", help="生成医学 QA 对")
    gen_p.add_argument("--tier", default="small", choices=["small", "medium", "large"],
                       help="数据集档位 (small/medium/large)")

    export_p = sub.add_parser("export", help="导出为指令微调格式")
    export_p.add_argument("--fmt", default="chatml", choices=["chatml", "alpaca"])

    sub.add_parser("stats", help="数据质量分析")

    sample_p = sub.add_parser("sample", help="预览文档内容")
    sample_p.add_argument("dir", nargs="?", default="L1_卫健委官方规范")

    sub.add_parser("all", help="全流程 (generate + stats + export)")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(tier=args.tier)
    elif args.command == "stats":
        cmd_stats()
    elif args.command == "export":
        cmd_export(args.fmt)
    elif args.command == "sample":
        cmd_sample(args.dir)
    elif args.command == "all":
        cmd_generate(tier="small")
        cmd_stats()
        cmd_export("chatml")
        cmd_export("alpaca")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
