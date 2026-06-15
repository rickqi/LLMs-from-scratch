"""
推理脚本: 公司规章制度文本生成
============================
用法:
  python generate.py --model_dir ./output/best_model
  python generate.py --model_dir ./output/best_model --interactive
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen3-0.6B"
TEST_PROMPTS = [
    "制度名称：",
    "适用范围：",
    "管理职责：",
    "操作流程：",
    "违规处理：",
    "审批权限：",
    "监督检查：",
    "附则：",
]


def load_model(model_dir: str):
    logger.info(f"加载基座模型: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": torch.device("cuda" if torch.cuda.is_available() else "cpu")},
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200, **kw):
    defaults = dict(max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=0.7, top_p=0.9, repetition_penalty=1.05)
    defaults.update(kw)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, **defaults)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def run_benchmark(model, tokenizer):
    logger.info(f"\n{'='*60}")
    logger.info("  公司规章制度生成评估")
    logger.info(f"{'='*60}\n")
    for prompt in TEST_PROMPTS:
        output = generate(model, tokenizer, prompt)
        new_content = output[len(prompt):].strip()
        print(f"【{prompt}】")
        print(f"  {new_content[:300]}")
        print()


def interactive(model, tokenizer):
    logger.info("\n交互模式 (输入 quit 退出)\n")
    while True:
        prompt = input(">>> ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue
        output = generate(model, tokenizer, prompt)
        print(f"\n{output[len(prompt):].strip()}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir)
    if args.prompt:
        output = generate(model, tokenizer, args.prompt, args.max_new_tokens)
        print(f"Prompt: {args.prompt}")
        print(f"生成: {output[len(args.prompt):].strip()}")
    elif args.interactive:
        interactive(model, tokenizer)
    else:
        run_benchmark(model, tokenizer)


if __name__ == "__main__":
    main()
