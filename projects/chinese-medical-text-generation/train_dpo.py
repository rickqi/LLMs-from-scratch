#!/usr/bin/env python3
"""
DPO 偏好对齐训练 — Qwen3-0.6B + LoRA

将 DPO 损失直接作用于 LoRA 适配器，对齐医学 QA 输出质量。
"""

import json, torch, logging, argparse, time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class DPODataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.samples = []
        for item in data:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            # Tokenize prompt+chosen
            c_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{chosen}<|im_end|>"
            r_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{rejected}<|im_end|>"

            c_tok = tokenizer(c_text, truncation=True, max_length=max_length, return_tensors="pt")
            r_tok = tokenizer(r_text, truncation=True, max_length=max_length, return_tensors="pt")

            self.samples.append({
                "chosen_input_ids": c_tok["input_ids"][0],
                "chosen_attention_mask": c_tok["attention_mask"][0],
                "rejected_input_ids": r_tok["input_ids"][0],
                "rejected_attention_mask": r_tok["attention_mask"][0],
            })

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def dpo_loss(model, ref_model, batch, beta=0.1):
    """DPO loss: -log σ(β * (log πθ(c) - log πref(c) - log πθ(r) + log πref(r)))"""

    def get_logprob(model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        # per-token log probs = -loss * sequence_length
        loss = outputs.loss
        return -loss * attention_mask.sum()

    with torch.no_grad():
        logp_ref_chosen = get_logprob(ref_model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
        logp_ref_rejected = get_logprob(ref_model, batch["rejected_input_ids"], batch["rejected_attention_mask"])

    logp_chosen = get_logprob(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
    logp_rejected = get_logprob(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])

    diff = logp_chosen - logp_ref_chosen - logp_rejected + logp_ref_rejected
    loss = -torch.nn.functional.logsigmoid(beta * diff).mean()
    return loss


def collate_fn(batch):
    """Pad sequences in batch"""
    max_c = max(b["chosen_input_ids"].size(0) for b in batch)
    max_r = max(b["rejected_input_ids"].size(0) for b in batch)

    c_ids = torch.stack([torch.nn.functional.pad(b["chosen_input_ids"], (0, max_c - b["chosen_input_ids"].size(0)), value=0) for b in batch])
    c_mask = torch.stack([torch.nn.functional.pad(b["chosen_attention_mask"], (0, max_c - b["chosen_attention_mask"].size(0)), value=0) for b in batch])
    r_ids = torch.stack([torch.nn.functional.pad(b["rejected_input_ids"], (0, max_r - b["rejected_input_ids"].size(0)), value=0) for b in batch])
    r_mask = torch.stack([torch.nn.functional.pad(b["rejected_attention_mask"], (0, max_r - b["rejected_attention_mask"].size(0)), value=0) for b in batch])

    return {
        "chosen_input_ids": c_ids.to(DEVICE),
        "chosen_attention_mask": c_mask.to(DEVICE),
        "rejected_input_ids": r_ids.to(DEVICE),
        "rejected_attention_mask": r_mask.to(DEVICE),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="./output_inst_v3/best_model")
    parser.add_argument("--preference_data", default="./data/dpo_pairs.json")
    parser.add_argument("--output_dir", default="./output_dpo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B", help="Base model name or local path")
    args = parser.parse_args()

    # Load base model + tokenizer
    logger.info(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=True
    )

    # Load LoRA adapter
    if Path(args.base_model).exists():
        model = PeftModel.from_pretrained(base_model, args.base_model, is_trainable=True)
        logger.info(f"LoRA adapter loaded: {args.base_model}")
    else:
        logger.warning("No LoRA adapter found, creating new one")
        lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(base_model, lora_config)

    model.to(DEVICE)

    # Reference model (frozen copy of starting point)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    if Path(args.base_model).exists():
        ref_model = PeftModel.from_pretrained(ref_model, args.base_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Load preference data
    with open(args.preference_data, encoding="utf-8") as f:
        pref_data = json.load(f)
    logger.info(f"Preference pairs: {len(pref_data)}")

    dataset = DPODataset(pref_data, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Optimizer (only LoRA params)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Training
    logger.info(f"DPO training: {args.epochs} epochs, lr={args.lr}, beta={args.beta}")
    model.train()
    t0 = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch in loader:
            opt.zero_grad()
            loss = dpo_loss(model, ref_model, batch, beta=args.beta)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        logger.info(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}")

    elapsed = time.time() - t0
    logger.info(f"Training complete: {elapsed:.0f}s")

    # Save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model saved: {args.output_dir}")


if __name__ == "__main__":
    main()
