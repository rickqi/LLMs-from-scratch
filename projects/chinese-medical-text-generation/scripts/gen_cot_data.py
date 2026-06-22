#!/usr/bin/env python3
"""
CoT (Chain-of-Thought) 思维链训练数据生成
==========================================
为医学 QA 数据添加推理步骤，生成 Qwen3 原生 <think> 格式训练数据。

用法:
  python scripts/gen_cot_data.py                  # 从现有 QA 生成 CoT 数据
  python scripts/gen_cot_data.py --count 200       # 指定生成数量
  python scripts/gen_cot_data.py --dry-run         # 预览不调用 API
"""

import json, os, sys, re, random, argparse, time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

# DeepSeek config (same as gen_qa_docsearch)
def _load_api_key():
    for src in [os.environ.get("DEEPSEEK_API_KEY",""),
                (Path("/mnt/d/docs/doc-search/.env").read_text() if Path("/mnt/d/docs/doc-search/.env").exists() else "")]:
        if not src: continue
        for line in src.splitlines():
            if "DEEPSEEK_API_KEY=" in line:
                return line.split("=",1)[1].strip().strip('"').strip("'")
    return ""

DEEPSEEK_API_KEY = _load_api_key()

COT_PROMPT = """你是一位医学教育专家。请为以下医学问答添加详细的推理过程。

## 原始问答
问题：{question}
答案：{answer}

## 要求
1. 在 <think> 标签中写出推理过程（200-400字）
2. 推理内容应包括：
   - 识别问题的核心医学概念
   - 回顾相关的诊断标准/治疗指南
   - 分析为什么该答案是正确的
   - 如有鉴别诊断，说明排除其他选项的理由
3. 推理后输出精炼的最终答案（在 </think> 之后）
4. 使用专业医学术语

## 输出格式
<think>
[详细的医学推理过程]
</think>

[精炼后的最终答案]
"""

def generate_cot(question: str, answer: str, client, model: str) -> dict | None:
    """用 DeepSeek 为单个 QA 生成 CoT 推理"""
    prompt = COT_PROMPT.format(question=question, answer=answer)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=1500,
        )
        raw = response.choices[0].message.content.strip()

        # Parse <think>...</think> and final answer
        think_match = re.search(r'<think>\s*(.*?)\s*</think>', raw, re.DOTALL)
        if think_match:
            think = think_match.group(1).strip()
            answer_part = raw[think_match.end():].strip()
        else:
            think = raw[:400].strip()
            answer_part = raw[400:].strip() if len(raw) > 400 else raw

        return {"think": think, "answer": answer_part}
    except Exception as e:
        print(f"  ⚠ API error: {e}")
        return None

def load_qa_samples(count: int) -> list[dict]:
    """从现有 ChatML 加载 QA 样本"""
    path = DOCS_DIR / "med_instruction_chatml.json"
    if not path.exists():
        print(f"❌ {path} not found")
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("data", [])
    random.shuffle(items)

    samples = []
    seen = set()
    for item in items:
        msgs = item.get("messages", [])
        q = next((m["content"] for m in msgs if m["role"] == "user"), "")
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        if q and a and q[:60] not in seen:
            samples.append({"question": q, "answer": a})
            seen.add(q[:60])
        if len(samples) >= count:
            break
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=150, help="生成 CoT 数量")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = DOCS_DIR / "med_instruction_cot_chatml.json"

    # Load samples
    samples = load_qa_samples(args.count)
    print(f"Loaded {len(samples)} QA samples")

    if args.dry_run:
        for i, s in enumerate(samples[:5], 1):
            print(f"\n[{i}] Q: {s['question'][:60]}...")
            print(f"    A: {s['answer'][:80]}...")
        return

    # Generate CoT
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    model = "deepseek-chat"

    SYSTEM = "你是一位中文医学助手，基于权威诊疗指南提供专业信息。请先推理再回答。"

    cot_items = []
    success = 0
    t0 = time.time()

    for i, s in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {s['question'][:50]}...", end=" ", flush=True)
        result = generate_cot(s["question"], s["answer"], client, model)
        if result:
            cot_text = f"<think>\n{result['think']}\n</think>\n\n{result['answer']}"
            cot_items.append({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": s["question"]},
                    {"role": "assistant", "content": cot_text},
                ],
                "metadata": {"format": "cot", "source": "deepseek_cot_generation"},
            })
            success += 1
            print(f"✅ {len(result['think'])}字推理")
        else:
            print("❌")

        # INCREMENTAL SAVE every 20 items
        if i % 20 == 0:
            output = {"format": "chatml_cot", "total": len(cot_items), "generated_at": datetime.now().isoformat(), "data": cot_items}
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"  💾 已保存 {len(cot_items)} 条 (checkpoint)")

    # Final save
    output = {"format": "chatml_cot", "total": len(cot_items), "generated_at": datetime.now().isoformat(), "data": cot_items}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ {success}/{len(samples)} CoT pairs → {output_path}")
    print(f"⏱ {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
