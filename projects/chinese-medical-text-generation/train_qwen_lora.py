"""
Qwen3-0.6B + LoRA 微调: 中文医学诊疗指南文本生成
================================================
用法:
  python train_qwen_lora.py --data_dir ./data --output_dir ./output

依赖:
  pip install transformers peft datasets accelerate wandb
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import time
import logging
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_LENGTH = 512
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 5
LEARNING_RATE = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


class MedicalTextDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        segments = text.split("===SEP===")
        self.examples = []
        for seg in segments:
            seg = seg.strip()
            if len(seg) < 10:
                continue
            enc = tokenizer(
                seg,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
            input_ids = enc["input_ids"]
            labels = input_ids.copy()
            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(ids) for ids in input_ids],
        batch_first=True,
        padding_value=0,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(ids) for ids in labels],
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = (input_ids != 0).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            total_loss += outputs.loss.item()
            num_batches += 1
    model.train()
    return total_loss / max(num_batches, 1)


def generate_sample(model, tokenizer, prompt: str, device, max_new_tokens=80):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    model.train()
    return generated


def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B + LoRA 医学文本微调")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录 (含 train.txt / val.txt)")
    parser.add_argument("--output_dir", type=str, default="./output", help="模型保存目录")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="批次大小")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="学习率")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="最大序列长度")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info(f"使用设备: {device}")

    # 1. 加载 tokenizer 和模型
    logger.info(f"加载模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    model = model.to(device)

    # 2. 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 3. 加载数据
    logger.info(f"加载数据: {data_dir}")
    tokenizer.model_max_length = args.max_length
    train_dataset = MedicalTextDataset(str(data_dir / "train.txt"), tokenizer, args.max_length)
    val_dataset = MedicalTextDataset(str(data_dir / "val.txt"), tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    logger.info(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    logger.info(f"训练批次/epoch: {len(train_loader)}, 验证批次: {len(val_loader)}")

    # 4. 优化器 & 调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )

    # 5. 训练循环
    logger.info("开始训练...")
    global_step = 0
    best_val_loss = float("inf")
    train_loss_log = []
    val_loss_log = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 定期评估
                if global_step % 10 == 0:
                    val_loss = evaluate(model, val_loader, device)
                    val_loss_log.append((global_step, val_loss))

                    avg_train_loss = epoch_loss / (step + 1)
                    train_loss_log.append((global_step, avg_train_loss))

                    elapsed = time.time() - start_time
                    logger.info(
                        f"Epoch {epoch}/{args.epochs} | "
                        f"Step {global_step:06d} | "
                        f"Train loss: {avg_train_loss:.4f} | "
                        f"Val loss: {val_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"Time: {elapsed:.0f}s"
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model.save_pretrained(output_dir / "best_model")
                        tokenizer.save_pretrained(output_dir / "best_model")
                        logger.info(f"  → 保存最佳模型 (val_loss={val_loss:.4f})")

        # 每个 epoch 生成样本
        for prompt in ["临床表现：", "诊断依据：", "治疗方案：", "预后判断："]:
            sample = generate_sample(model, tokenizer, prompt, device)
            logger.info(f"\n[生成样本 | prompt: {prompt}]\n{sample}\n")

    # 6. 保存最终模型和训练记录
    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")
    logger.info(f"最终模型保存到: {output_dir / 'final_model'}")

    records = {
        "train_loss": train_loss_log,
        "val_loss": val_loss_log,
        "best_val_loss": best_val_loss,
        "total_time_seconds": time.time() - start_time,
        "config": {
            "model": MODEL_NAME,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "max_length": args.max_length,
        },
    }
    with open(output_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"训练日志保存到: {output_dir / 'training_log.json'}")
    logger.info(f"训练完成! 总耗时: {(time.time()-start_time)/60:.2f} 分钟")


if __name__ == "__main__":
    main()
