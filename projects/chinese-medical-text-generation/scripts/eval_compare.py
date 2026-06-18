"""
多模型评测对比脚本
==================
用法:
# 评测 0.6B 基线
python scripts/eval_compare.py --model_dir ./output_inst_v3/best_model

# 评测 1.7B (训练完成后)
python scripts/eval_compare.py --model_dir ./output_17b_inst/best_model --base_model /home/models/ms_cache/Qwen/Qwen3-1___7B

# 两者对比
python scripts/eval_compare.py --compare --baseline ./output_inst_v3/best_model --target ./output_17b_inst/best_model --target_base /home/models/ms_cache/Qwen/Qwen3-1___7B
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# 9 道标准评测题（对应 REPORT.md 评测表 + 甲状腺专项）
EVAL_QUESTIONS = [
    "胃癌的典型临床表现有哪些？",
    "糖尿病患者的饮食管理原则是什么？",
    "什么是医院感染的预防控制措施？",
    "高血压的药物治疗原则有哪些？",
    "急性心肌梗死的诊断标准是什么？",
    "肺癌的TNM分期标准是什么？",
    "手术后需要观察哪些并发症？",
    "请对比CT和MRI在肿瘤分期中的优缺点。",
    "甲状腺切除手术对身体有什么影响？晕倒风险如何？",
]


def load_model(model_dir: str, base_model: str):
    """加载基座模型 + LoRA adapter"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    logger.info(f"基座模型: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, model_dir)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, question: str, **kwargs) -> dict:
    """生成单题回答，返回 dict"""
    import torch

    defaults = dict(
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    defaults.update(kwargs)

    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **defaults)
    elapsed = time.time() - t0

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # 提取 assistant 部分 (去掉 ChatML 模板)
    answer = full_output
    if "assistant" in full_output.lower():
        parts = full_output.split("assistant", 1)
        if len(parts) > 1:
            answer = parts[1].strip()
            if answer.startswith("\n"):
                answer = answer[1:]

    new_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]
    return {
        "question": question,
        "answer": answer,
        "new_tokens": new_tokens,
        "time_seconds": round(elapsed, 2),
    }


def run_eval(model_dir: str, base_model: str, output_path: Optional[str] = None, **gen_kwargs) -> list[dict]:
    """运行完整评测"""
    model, tokenizer = load_model(model_dir, base_model)

    results = []
    logger.info(f"\n{'='*60}")
    logger.info(f"  评测: {model_dir}")
    logger.info(f"  基座: {base_model}")
    logger.info(f"{'='*60}\n")

    for i, q in enumerate(EVAL_QUESTIONS, 1):
        logger.info(f"[{i}/9] {q[:40]}...")
        r = generate_response(model, tokenizer, q, **gen_kwargs)
        r["index"] = i
        print(f"\n--- Q{i} ---")
        print(f"Q: {q}")
        print(f"A: {r['answer'][:500]}")
        print(f"Tokens: {r['new_tokens']} | Time: {r['time_seconds']}s\n")
        results.append(r)

    # 保存 JSON
    if output_path is None:
        model_name = Path(model_dir).parent.name
        output_path = f"eval_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_dir": model_dir,
        "base_model": base_model,
        "timestamp": datetime.now().isoformat(),
        "gen_kwargs": gen_kwargs,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"结果已保存: {output_path}")

    return results


def compare(baseline_path: str, target_path: str):
    """对比两个模型的 JSON 结果，打印差异摘要"""
    import json

    def load_results(path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {r["question"]: r for r in data["results"]}

    baseline = load_results(baseline_path)
    target = load_results(target_path)

    print("\n" + "=" * 60)
    print("  模型对比")
    print("=" * 60)

    for i, q in enumerate(EVAL_QUESTIONS, 1):
        b = baseline.get(q)
        t = target.get(q)
        if not b or not t:
            print(f"\n[Q{i}] {q[:50]}... — 缺失数据，跳过")
            continue

        print(f"\n{'─'*60}")
        print(f"[Q{i}] {q}")
        print(f"{'─'*60}")
        print(f"基线 ({baseline_path}): {b['new_tokens']} tokens, {b['time_seconds']}s")
        print(f"  {b['answer'][:300]}")
        print()
        print(f"目标 ({target_path}): {t['new_tokens']} tokens, {t['time_seconds']}s")
        print(f"  {t['answer'][:300]}")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="多模型医学 QA 评测")
    parser.add_argument("--model_dir", type=str, default=None, help="LoRA adapter 目录")
    parser.add_argument("--base_model", type=str, default=None,
                        help="基座模型路径 (默认自动选择)")
    parser.add_argument("--output", type=str, default=None, help="结果保存路径")
    parser.add_argument("--compare", action="store_true", help="对比两个已有评测结果")
    parser.add_argument("--baseline", type=str, default=None, help="基线模型 JSON 结果路径")
    parser.add_argument("--target", type=str, default=None, help="目标模型 JSON 结果路径")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    args = parser.parse_args()

    if args.compare:
        if not args.baseline or not args.target:
            parser.error("--compare 需要 --baseline 和 --target")
        compare(args.baseline, args.target)
        return

    if not args.model_dir:
        parser.error("需要 --model_dir 或 --compare")

    # 自动推断 base_model
    base_model = args.base_model
    if base_model is None:
        # 从 adapter_config.json 读取
        import json
        cfg = json.loads((Path(args.model_dir) / "adapter_config.json").read_text())
        base_model = cfg.get("base_model_name_or_path", "Qwen/Qwen3-0.6B")

    gen_kwargs = dict(temperature=args.temperature, max_new_tokens=args.max_new_tokens)
    run_eval(args.model_dir, base_model, args.output, **gen_kwargs)


if __name__ == "__main__":
    main()
