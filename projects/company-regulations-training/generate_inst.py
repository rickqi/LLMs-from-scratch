"""
指令模型交互式推理: ChatML 格式
用法: python generate_inst.py --model_dir ./output_inst_v2/best_model
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-0.6B"

def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": torch.device("cuda" if torch.cuda.is_available() else "cpu")},
    )
    model = PeftModel.from_pretrained(base, model_dir)
    model.eval()
    return model, tokenizer

def ask(model, tokenizer, question, max_new=300):
    messages = [
        {"role": "system", "content": "你是一位公司制度管理助手，基于公司规章制度提供准确的信息。回答应当条理清晰、引用具体条款。"},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=True,
            temperature=0.7, top_p=0.9,
        )
    new_token_ids = output_ids[0][input_len:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir)
    print(f"\n模型已加载: {args.model_dir}")
    print("输入问题 (quit 退出)\n")

    if args.prompt:
        answer = ask(model, tokenizer, args.prompt, args.max_new_tokens)
        print(f"Q: {args.prompt}")
        print(f"A: {answer}")
        return

    while True:
        try:
            q = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        answer = ask(model, tokenizer, q, args.max_new_tokens)
        print(f"\n{answer}\n")

if __name__ == "__main__":
    main()
