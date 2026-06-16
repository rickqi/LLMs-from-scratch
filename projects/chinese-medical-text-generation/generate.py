"""
推理/评估脚本: 加载微调后的 Qwen3-0.6B + LoRA 模型, 生成中文医学文本
=====================================================================
用法:
  python generate.py --model_dir ./output/best_model
  python generate.py --model_dir ./output/best_model --interactive
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen3-0.6B"
TEST_PROMPTS = [
    "临床表现：",
    "诊断依据：",
    "治疗方案：",
    "预后判断：",
    "鉴别诊断：",
    "并发症：",
    "用药原则：",
    "出院标准：",
]


def load_model(model_dir: str):
    logger.info(f"加载基座模型: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200, instruct: bool = False, **gen_kwargs):
    defaults = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
    )
    defaults.update(gen_kwargs)

    device = next(model.parameters()).device
    if instruct:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **defaults)

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated


def run_benchmark(model, tokenizer, prompts: list[str], instruct: bool = False):
    logger.info(f"\n{'='*60}")
    logger.info("  医学文本生成评估")
    logger.info(f"{'='*60}\n")

    for prompt in prompts:
        output = generate(model, tokenizer, prompt, instruct=instruct)
        new_content = output[len(prompt):].strip()
        print(f"【{prompt}】")
        print(f"  {new_content[:300]}")
        print()


def interactive_mode(model, tokenizer, instruct: bool = False):
    logger.info("\n进入交互模式 (输入 'quit' 退出)\n")
    while True:
        prompt = input(">>> 输入提示词: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue
        output = generate(model, tokenizer, prompt, instruct=instruct)
        new_content = output[len(prompt):].strip()
        print(f"\n生成结果:\n{new_content}\n")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B + LoRA 医学文本生成推理")
    parser.add_argument("--model_dir", type=str, required=True, help="微调后的 LoRA 权重目录")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--instruct", action="store_true", help="使用 ChatML 指令格式 (用于指令微调模型)")
    parser.add_argument("--prompt", type=str, default=None, help="单次生成提示词")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir)

    if args.prompt:
        output = generate(model, tokenizer, args.prompt, args.max_new_tokens, instruct=args.instruct, temperature=args.temperature)
        print(f"Prompt: {args.prompt}")
        print(f"生成:\n{output}")
        print()
    elif args.interactive:
        interactive_mode(model, tokenizer, instruct=args.instruct)
    else:
        run_benchmark(model, tokenizer, TEST_PROMPTS, instruct=args.instruct)


if __name__ == "__main__":
    main()
