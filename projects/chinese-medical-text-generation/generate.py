"""
推理/评估脚本: 加载微调后的 Qwen3-0.6B + LoRA 模型, 生成中文医学文本
======================================================================
用法:
  python generate.py --model_dir ./output/best_model
  python generate.py --model_dir ./output_inst_v1/best_model --instruct
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

# 从统一模块导入（不要在此文件直接修改）
from test_questions import CONT_PROMPTS as TEST_PROMPTS
from test_questions import INST_QUESTIONS as INST_TEST_PROMPTS


def load_model(model_dir: str, base_model: str = None):
    if base_model is None:
        base_model = BASE_MODEL
    logger.info(f"加载基座模型: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200,
             instruct: bool = False, **gen_kwargs):
    defaults = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
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
        print(f"【{prompt}】")
        print(f"  {output[:400]}")
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
        print(f"\n生成结果:\n{output}\n")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B + LoRA 医学文本生成推理")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None, help="基座模型名称 (默认 Qwen3-0.6B)")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--instruct", action="store_true", help="使用 ChatML 指令格式")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚 (越大越不重复)")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir, args.base_model)

    if args.prompt:
        output = generate(model, tokenizer, args.prompt, args.max_new_tokens,
                         instruct=args.instruct, temperature=args.temperature,
                         repetition_penalty=args.repetition_penalty)
        print(f"Prompt: {args.prompt}")
        print(f"生成:\n{output}")
    elif args.interactive:
        interactive_mode(model, tokenizer, instruct=args.instruct)
    else:
        prompts = INST_TEST_PROMPTS if args.instruct else TEST_PROMPTS
        run_benchmark(model, tokenizer, prompts, instruct=args.instruct)


if __name__ == "__main__":
    main()
