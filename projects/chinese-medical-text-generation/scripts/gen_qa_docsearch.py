#!/usr/bin/env python3
"""
doc-search 医学知识库集成 — 高质量 QA 生成
==========================================
用 doc-search 的 BM25 检索 + DeepSeek LLM 生成答案，产出比纯 LLM 凭空生成更高质量的 QA。

工作流:
  1. BM25 检索 → 找到最相关的源文档段落
  2. DeepSeek 基于检索结果生成答案（有原文依据，非凭空）
  3. 输出 ChatML 格式，source_text 可追溯

用法:
  python scripts/gen_qa_docsearch.py                    # 用预设问题列表
  python scripts/gen_qa_docsearch.py --query "问题"      # 单问题
  python scripts/gen_qa_docsearch.py --input questions.txt  # 从文件读
"""

import json, os, sys, re, hashlib, argparse, logging, subprocess
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCSEARCH_ROOT = Path("/mnt/d/docs/doc-search")
DOCSEARCH_PYTHON = DOCSEARCH_ROOT / ".venv" / "Scripts" / "python.exe"

# 医学索引路径（doc-search 的 medica 4 个索引）
MEDICA_INDEXES = (
    r"D:\docs\raw\medica\L1_卫健委官方规范\index,"
    r"D:\docs\raw\medica\L2_卫生行业标准_WS\index,"
    r"D:\docs\raw\medica\L3_中华医学会临床诊疗指南丛书\index,"
    r"D:\docs\raw\medica\L4_中华医学会临床技术操作规范\index"
)

MEDICA_RAW_DIRS = [
    r"D:\docs\raw\medica\L1_卫健委官方规范",
    r"D:\docs\raw\medica\L2_卫生行业标准_WS",
    r"D:\docs\raw\medica\L3_中华医学会临床诊疗指南丛书",
    r"D:\docs\raw\medica\L4_中华医学会临床技术操作规范",
]

# LLM 配置 — 自动从多个来源读取 API key
def _load_deepseek_key():
    """自动检测 DeepSeek API key"""
    # 1. 环境变量
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key
    # 2. doc-search .env
    doc_env = Path("/mnt/d/docs/doc-search/.env")
    if doc_env.exists():
        for line in doc_env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    # 3. 本项目 .env
    proj_env = PROJECT_ROOT / ".env"
    if proj_env.exists():
        for line in proj_env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

DEEPSEEK_API_KEY = _load_deepseek_key()
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 预设问题列表（聚焦薄弱领域）
PRESET_QUESTIONS = [
    # 用药/化疗方案
    "抗肿瘤药物化疗后骨髓抑制的分级标准和处理原则",
    "铂类药物在肿瘤化疗中的剂量调整原则和不良反应管理",
    "肿瘤靶向治疗和免疫治疗药物的选择依据和生物标志物检测",
    # 感染控制
    "多重耐药菌感染患者的隔离措施和消毒技术规范",
    "手术部位感染的预防控制措施和监测标准",
    "医院感染暴发的定义、报告流程和应急处置",
    # 术后并发症
    "胃癌根治术后吻合口漏的早期识别和处理方案",
    "术后深静脉血栓的预防策略和抗凝管理",
    "术后疼痛管理的多模式镇痛方案",
    # 诊断标准
    "肿瘤标志物在恶性肿瘤诊断和复发监测中的临床应用价值",
    "液体活检在肿瘤早期诊断和微小残留病灶监测中的应用",
    "多学科讨论MDT在肿瘤诊疗决策中的流程和价值",
    # 影像对比
    "增强CT和MRI在肝脏肿瘤鉴别诊断中的优劣对比",
    "PET-CT在恶性肿瘤远处转移评估中的适应症和局限性",
    # 预后随访
    "恶性肿瘤患者术后随访的时间间隔和检查项目规范",
    "肿瘤复发转移的高危因素和早期预警指标",
]

QA_GEN_PROMPT = """你是一位资深临床医学专家。请根据以下检索到的医学文献内容，生成一个高质量的问答题。

## 用户问题
{question}

## 检索到的相关文献（从医学知识库中 BM25 检索获得）
{context}

## 要求
1. 答案必须严格基于上述检索到的文献内容，不得编造
2. 使用专业准确的医学术语
3. 结构化组织答案（使用标题、分点）
4. 如果文献中包含具体数字、标准、指南名称，请准确引用
5. 答案长度：300-800字

## 输出格式
严格按以下 JSON 输出：
```json
{{
  "question": "{question}",
  "answer": "基于文献的结构化答案",
  "source_text": "答案依据的核心原文片段（至少100字）",
  "difficulty": "medium",
  "query_type": "factual",
  "keywords": ["关键词1", "关键词2", "关键词3"]
}}
```"""


def run_bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """通过 doc-search Python API 执行 BM25 搜索"""
    searcher_code = f'''
import sys, json
from pathlib import Path
sys.path.insert(0, r"D:\\\\docs\\\\doc-search")
from src.search.multi_index import MultiIndexSearcher

paths = [
    Path(r"D:\\\\docs\\\\raw\\\\medica\\\\L1_卫健委官方规范\\\\index"),
    Path(r"D:\\\\docs\\\\raw\\\\medica\\\\L2_卫生行业标准_WS\\\\index"),
    Path(r"D:\\\\docs\\\\raw\\\\medica\\\\L3_中华医学会临床诊疗指南丛书\\\\index"),
    Path(r"D:\\\\docs\\\\raw\\\\medica\\\\L4_中华医学会临床技术操作规范\\\\index"),
]
searcher = MultiIndexSearcher(paths)
result = searcher.search("{query}", limit={top_k})
output = []
for r in result.results[:{top_k}]:
    output.append({{
        "score": r.score,
        "title": r.title,
        "source": str(r.source_path),
        "snippet": (getattr(r, "snippet", "") or "")[:300],
    }})
print("JSON_START")
print(json.dumps(output, ensure_ascii=False))
print("JSON_END")
'''
    try:
        result = subprocess.run(
            [str(DOCSEARCH_PYTHON), "-c", searcher_code],
            capture_output=True, text=True, timeout=30,
            cwd=str(DOCSEARCH_ROOT),
            encoding="gbk", errors="replace",
        )
        output = result.stdout
        match = re.search(r'JSON_START\n(.*?)\nJSON_END', output, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception as e:
        logger.warning(f"BM25 search failed: {e}")
    return []


def read_document(source_path: str, max_chars: int = 3000) -> str:
    """读取检索到的文档内容（处理 Windows→Linux 路径翻译）"""
    # doc-search returns Windows paths like:
    #   "肿瘤\中国急性胰腺炎诊治指南2021.pdf.md"
    # We need to find this under /mnt/d/docs/raw/medica/{L1|L2|L3|L4}/
    source_path = source_path.replace("\\", "/")
    for raw_dir in MEDICA_RAW_DIRS:
        # Translate Windows path to Linux
        linux_dir = raw_dir.replace("\\", "/").replace("D:", "/mnt/d")
        full = Path(linux_dir) / source_path
        if full.exists():
            try:
                text = full.read_text(encoding="utf-8", errors="replace")
                text = re.sub(r'<[^>]+>', '', text)
                text = re.sub(r'\n{4,}', '\n\n', text)
                if len(text) > max_chars:
                    text = text[:max_chars]
                return text
            except Exception:
                pass
    return ""


def filter_search_results(results: list[dict]) -> list[dict]:
    """过滤和排序搜索结果：优先匹配度高、内容丰富的文档"""
    filtered = []
    for r in results:
        score = r.get("score", 0)
        title = r.get("title", "")
        source = r.get("source_path", r.get("source", ""))
        
        # 过滤明显不相关的文档
        skip_keywords = ["版本表", "封面", "目录", "前言"]
        if any(kw in title for kw in skip_keywords):
            continue
        
        # 优先级：分数高的 + 标题匹配度高的
        filtered.append({
            "score": score,
            "title": title,
            "source": source,
            "snippet": r.get("snippet", r.get("content", ""))[:300],
        })
    
    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered[:5]


def generate_qa_with_context(question: str, search_results: list[dict]) -> dict:
    """基于检索上下文，用 DeepSeek 生成高质量 QA"""
    # Read top 3 document contents
    contexts = []
    source_texts = []
    for r in search_results[:3]:
        content = read_document(r["source"])
        if content:
            contexts.append(f"### {r['title']}\n{content[:2000]}")
            source_texts.append(content[:500])

    if not contexts:
        logger.warning(f"No readable documents found for: {question[:50]}...")
        return None

    context_str = "\n\n---\n\n".join(contexts)
    prompt = QA_GEN_PROMPT.format(question=question, context=context_str)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()

        # Extract JSON
        json_match = re.search(r'```json\s*(.*?)```', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1)
        elif not raw.startswith("{"):
            json_match2 = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match2:
                raw = json_match2.group(0)

        result = json.loads(raw)
        # Add source tracking
        result["source_docs"] = [
            {"title": r["title"], "path": r["source"], "score": r["score"]}
            for r in search_results[:3]
        ]
        result["generation_method"] = "bm25_retrieval + deepseek"
        return result

    except Exception as e:
        logger.warning(f"QA generation failed: {e}")
        return None


def convert_to_chatml(qa_item: dict) -> dict:
    """转换为 ChatML 训练格式"""
    SYSTEM_PROMPT = (
        "你是一位中文医学助手，基于权威诊疗指南、行业标准和临床操作规范提供专业的医学信息。"
        "你的回答应当准确、结构化，并引用具体的医学标准或指南来源。"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": qa_item["question"]},
            {"role": "assistant", "content": qa_item["answer"]},
        ],
        "metadata": {
            "source_docs": qa_item.get("source_docs", []),
            "difficulty": qa_item.get("difficulty", "medium"),
            "query_type": qa_item.get("query_type", "factual"),
            "generation_method": qa_item.get("generation_method", "bm25_retrieval"),
            "keywords": qa_item.get("keywords", []),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="doc-search 医学知识库 QA 生成")
    parser.add_argument("--query", type=str, default=None, help="单个查询")
    parser.add_argument("--input", type=str, default=None, help="从文件读取问题（每行一个）")
    parser.add_argument("--output", type=str, default=None, help="输出 ChatML 文件路径")
    parser.add_argument("--top-k", type=int, default=5, help="BM25 检索结果数")
    args = parser.parse_args()

    # 获取问题列表
    if args.query:
        questions = [args.query]
    elif args.input:
        questions = [l.strip() for l in open(args.input, encoding="utf-8") if l.strip()]
    else:
        questions = PRESET_QUESTIONS

    # 输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / "docs" / "med_instruction_docsearch_chatml.json"

    logger.info(f"问题数: {len(questions)}")
    logger.info(f"doc-search 索引: {MEDICA_INDEXES.count('index')} 个")
    logger.info(f"输出: {output_path}")

    all_chatml = []
    success = 0

    for i, q in enumerate(questions, 1):
        logger.info(f"\n[{i}/{len(questions)}] {q[:60]}...")

        # Step 1: BM25 search
        logger.info("  BM25 检索...")
        results = run_bm25_search(q, top_k=args.top_k)
        if not results:
            logger.warning("  无搜索结果，跳过")
            continue

        filtered = filter_search_results(results)
        logger.info(f"  找到 {len(filtered)} 个相关文档:")
        for r in filtered[:3]:
            logger.info(f"    - [{r['score']:.3f}] {r['title'][:50]}")

        # Step 2: Generate QA with context
        logger.info("  LLM 生成答案...")
        qa = generate_qa_with_context(q, filtered)
        if not qa:
            logger.warning("  生成失败，跳过")
            continue

        chatml = convert_to_chatml(qa)
        all_chatml.append(chatml)
        success += 1
        logger.info(f"  ✅ 成功 ({len(qa['answer'])} 字答案)")

    # Save
    output = {
        "format": "chatml",
        "total": len(all_chatml),
        "generated_at": datetime.now().isoformat(),
        "method": "docsearch_bm25_retrieval + deepseek_generation",
        "data": all_chatml,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"  完成: {success}/{len(questions)} 条成功")
    logger.info(f"  保存: {output_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
