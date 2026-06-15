"""
Qwen3-0.6B + LoRA 微调: 公司规章制度文本生成
=============================================
用法:
  python train_qwen_lora.py --data_dir ./data --output_dir ./output

环境: AMD Radeon 890M + ROCm 7.2 + PyTorch 2.9
依赖: pip install transformers peft datasets accelerate
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


# ============================================================
# 配置 (针对 AMD 890M 优化)
# ============================================================
MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_LENGTH = 512
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 5
LEARNING_RATE = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# 公司制度领域的生成测试提示词
EVAL_PROMPTS = [
    "制度名称：",
    "适用范围：",
    "管理职责：",
    "操作流程：",
    "违规处理：",
    "审批权限：",
    "监督检查：",
    "附则：",
]


# ============================================================
# 数据集
# ============================================================
class RegulationDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, stride: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride if stride is not None else max_length
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        segments = text.split("===SEP===")
        self.examples = []
        for seg in segments:
            seg = seg.strip()
            if len(seg) < 10:
                continue
            token_ids = tokenizer.encode(seg, add_special_tokens=False)
            if len(token_ids) <= max_length:
                self.examples.append({"input_ids": token_ids, "labels": token_ids.copy()})
            else:
                for start in range(0, len(token_ids) - max_length + 1, self.stride):
                    chunk = token_ids[start:start + max_length]
                    self.examples.append({"input_ids": chunk, "labels": chunk.copy()})
                if len(token_ids) % self.stride != 0:
                    last_chunk = token_ids[-max_length:]
                    self.examples.append({"input_ids": last_chunk, "labels": last_chunk.copy()})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(ids) for ids in input_ids],
        batch_first=True, padding_value=0,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(ids) for ids in labels],
        batch_first=True, padding_value=-100,
    )
    attention_mask = (input_ids != 0).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ============================================================
# 设备选择 (AMD GPU via ROCm)
# ============================================================
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"VRAM: {vram:.1f} GB")
        return device
    return torch.device("cpu")


# ============================================================
# 评估 & 生成
# ============================================================
def evaluate(model, val_loader, device, max_batches=30):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
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
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
        )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    model.train()
    return generated


# ============================================================
# 主训练函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B + LoRA 公司制度微调")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info(f"使用设备: {device}")

    # 1. 加载模型
    logger.info(f"加载模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map={"": device} if device.type == "cuda" else None,
    )

    # 2. LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 3. 数据
    logger.info(f"加载数据: {data_dir}")
    tokenizer.model_max_length = args.max_length
    train_ds = RegulationDataset(str(data_dir / "train.txt"), tokenizer, args.max_length)
    val_ds = RegulationDataset(str(data_dir / "val.txt"), tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn)
    logger.info(f"训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")
    logger.info(f"训练批次/epoch: {len(train_loader)}")

    # 4. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.05*total_steps), num_training_steps=total_steps)

    # 5. 断点恢复
    ckpt_path = output_dir / "checkpoint"
    global_step = 0
    best_val_loss = float("inf")
    train_log, val_log = [], []
    start_epoch = 1
    elapsed_offset = 0.0

    if ckpt_path.exists():
        logger.info(f"发现断点: {ckpt_path}")
        ckpt = torch.load(ckpt_path / "training_state.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        train_log = ckpt.get("train_loss_log", [])
        val_log = ckpt.get("val_loss_log", [])
        start_epoch = ckpt.get("epoch", 1)
        elapsed_offset = ckpt.get("elapsed", 0.0)
        logger.info(f"从 epoch {start_epoch}, step {global_step} 恢复 (best_val={best_val_loss:.4f})")

    # 6. 训练
    logger.info("开始训练...")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
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

            if (step + 1) % 50 == 0:
                elapsed = elapsed_offset + (time.time() - start_time)
                logger.info(
                    f"Ep {epoch}/{args.epochs} | Batch {step+1}/{len(train_loader)} | "
                    f"loss={loss.item()*GRADIENT_ACCUMULATION_STEPS:.4f} | {elapsed:.0f}s"
                )

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 20 == 0:
                    val_loss = evaluate(model, val_loader, device)
                    val_log.append((global_step, val_loss))
                    avg_train_loss = epoch_loss / (step + 1)
                    train_log.append((global_step, avg_train_loss))
                    elapsed = elapsed_offset + (time.time() - start_time)
                    logger.info(
                        f"Ep {epoch}/{args.epochs} | Step {global_step:06d} | "
                        f"Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model.save_pretrained(output_dir / "best_model")
                        tokenizer.save_pretrained(output_dir / "best_model")
                        logger.info(f"  -> 保存最佳模型 (val_loss={val_loss:.4f})")

        # Epoch 断点
        ckpt_path.mkdir(parents=True, exist_ok=True)
        elapsed = elapsed_offset + (time.time() - start_time)
        torch.save({
            "epoch": epoch + 1, "global_step": global_step,
            "best_val_loss": best_val_loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss_log": train_log, "val_loss_log": val_log,
            "elapsed": elapsed,
        }, ckpt_path / "training_state.pt")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        logger.info(f"  -> 断点已保存 (epoch {epoch})")

        # 每个 epoch 生成样本
        for prompt in EVAL_PROMPTS:
            sample = generate_sample(model, tokenizer, prompt, device)
            logger.info(f"\n[生成样本 | {prompt}]\n{sample}\n")

    # 保存最终模型
    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")

    records = {
        "train_loss": train_log, "val_loss": val_log,
        "best_val_loss": best_val_loss,
        "total_time_seconds": elapsed_offset + (time.time() - start_time),
        "config": {
            "model": MODEL_NAME, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "batch_size": args.batch_size, "epochs": args.epochs,
            "lr": args.lr, "max_length": args.max_length,
        },
    }
    with open(output_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"训练完成! 总耗时: {(time.time()-start_time)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
