#!/usr/bin/env python3
"""
DPO 偏好对齐训练 — Qwen3 + LoRA
================================
将 DPO 损失直接作用于 LoRA 适配器，对齐医学 QA 输出质量。

用法:
  # 0.6B
  python train_dpo.py --base_model ./output_inst_v3/best_model --preference_data ./data/dpo_pairs_filtered.json

  # 1.7B (batch=1 防双模型 OOM, beta=0.05 更稳定)
  python train_dpo.py --base_model ./output_17b_inst_v2/best_model \\
      --preference_data ./data/dpo_pairs_filtered.json \\
      --model_name /home/models/ms_cache/Qwen/Qwen3-1___7B \\
      --output_dir ./output_17b_dpo_v2 \\
      --batch_size 1 --beta 0.05 --epochs 3 --lr 5e-6

训练标准:
  - 内置长度过滤器: 自动剔除 chosen/rejected 长度差 >50% 的 pair
  - 早停: 每 epoch 用 val_loss 监控，patience=3 无改善停止
  - 坍塌检测: val_loss 突降至 <0.1 时自动终止
"""

import argparse, json, logging, os, time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class DPODataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, max_len_diff=0.5):
        self.samples = []
        skipped = 0
        for item in data:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            # 长度偏差过滤（防模式坍塌）
            cl, rl = len(chosen), len(rejected)
            if max(cl, rl) > 0 and abs(cl - rl) / max(cl, rl) > max_len_diff:
                skipped += 1
                continue

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
        if skipped:
            logger.warning(f"  长度过滤: 跳过 {skipped} 对 (长度差 >{max_len_diff*100:.0f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def dpo_loss(model, ref_model, batch, beta=0.1):
    """DPO loss: -log σ(β * (log πθ(c|p) - log πref(c|p) - log πθ(r|p) + log πref(r|p)))"""

    def get_logprob(m, input_ids, attention_mask):
        outputs = m(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        return -outputs.loss * attention_mask.sum()

    with torch.no_grad():
        logp_ref_c = get_logprob(ref_model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
        logp_ref_r = get_logprob(ref_model, batch["rejected_input_ids"], batch["rejected_attention_mask"])

    logp_c = get_logprob(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
    logp_r = get_logprob(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])

    diff = logp_c - logp_ref_c - logp_r + logp_ref_r
    acc = (diff > 0).float().mean().item()
    loss = -torch.nn.functional.logsigmoid(beta * diff).mean()
    return loss, acc


def collate_fn(batch):
    pad = torch.nn.functional.pad
    max_c = max(b["chosen_input_ids"].size(0) for b in batch)
    max_r = max(b["rejected_input_ids"].size(0) for b in batch)
    return {
        "chosen_input_ids": torch.stack([
            pad(b["chosen_input_ids"], (0, max_c - b["chosen_input_ids"].size(0)), value=0)
            for b in batch
        ]).to(DEVICE),
        "chosen_attention_mask": torch.stack([
            pad(b["chosen_attention_mask"], (0, max_c - b["chosen_attention_mask"].size(0)), value=0)
            for b in batch
        ]).to(DEVICE),
        "rejected_input_ids": torch.stack([
            pad(b["rejected_input_ids"], (0, max_r - b["rejected_input_ids"].size(0)), value=0)
            for b in batch
        ]).to(DEVICE),
        "rejected_attention_mask": torch.stack([
            pad(b["rejected_attention_mask"], (0, max_r - b["rejected_attention_mask"].size(0)), value=0)
            for b in batch
        ]).to(DEVICE),
    }


def evaluate(model, ref_model, val_loader, beta):
    """DPO val metric: loss + accuracy across val split"""
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            loss, acc = dpo_loss(model, ref_model, batch, beta=beta)
            total_loss += loss.item()
            total_acc += acc
            n += 1
    model.train()
    return total_loss / max(n, 1), total_acc / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="./output_inst_v3/best_model", help="LoRA adapter 路径")
    parser.add_argument("--preference_data", default="./data/dpo_pairs_filtered.json")
    parser.add_argument("--output_dir", default="./output_dpo")
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B", help="基座模型名称或本地路径")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.05, help="DPO beta (0.05 防坍塌, 0.1 强对齐)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="早停 patience (epoch 级)")
    parser.add_argument("--collapse_threshold", type=float, default=0.1, help="val_loss < 此值 = 坍塌")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载基座模型 + tokenizer ──
    logger.info(f"基座模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=DEVICE if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # ── 加载 LoRA adapter ──
    logger.info(f"LoRA adapter: {args.base_model}")
    model = PeftModel.from_pretrained(base_model, args.base_model, is_trainable=True)

    # ── Reference model (冻结, 共享基座) ──
    ref_base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=DEVICE if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(ref_base, args.base_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ── 加载数据 (内置长度过滤) ──
    with open(args.preference_data, encoding="utf-8") as f:
        all_data = json.load(f)
    logger.info(f"偏好数据: {len(all_data)} pairs")

    split_idx = int(len(all_data) * (1 - args.val_ratio))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    logger.info(f"训练: {len(train_data)}, 验证: {len(val_data)}")

    train_dataset = DPODataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = DPODataset(val_data, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    logger.info(f"有效训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    # ── 优化器 ──
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # ── 训练循环 (含早停+坍塌检测) ──
    logger.info(f"DPO 训练: {args.epochs} epochs, lr={args.lr}, beta={args.beta}, batch={args.batch_size}")
    logger.info(f"早停 patience={args.early_stopping_patience} epochs, 坍塌阈值={args.collapse_threshold}")

    model.train()
    t0 = time.time()
    best_val_loss = float("inf")
    best_acc = 0.0
    no_improve = 0
    train_loss_log = []
    val_loss_log = []

    for epoch in range(args.epochs):
        epoch_loss, epoch_acc, n = 0.0, 0.0, 0
        for batch in train_loader:
            opt.zero_grad()
            loss, acc = dpo_loss(model, ref_model, batch, beta=args.beta)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            epoch_acc += acc
            n += 1

        avg_loss = epoch_loss / max(n, 1)
        avg_acc = epoch_acc / max(n, 1)
        val_loss, val_acc = evaluate(model, ref_model, val_loader, beta=args.beta)

        train_loss_log.append((epoch + 1, avg_loss, avg_acc))
        val_loss_log.append((epoch + 1, val_loss, val_acc))

        elapsed = time.time() - t0
        logger.info(f"Epoch {epoch+1}/{args.epochs} | "
                     f"Train loss={avg_loss:.4f} acc={avg_acc:.3f} | "
                     f"Val loss={val_loss:.4f} acc={val_acc:.3f} | "
                     f"Best val={best_val_loss:.4f} | Time={elapsed:.0f}s")

        # 坍塌检测
        if val_loss < args.collapse_threshold and avg_loss < args.collapse_threshold:
            logger.error(f"  ⚠ 模型坍塌! val_loss={val_loss:.4f} < {args.collapse_threshold} — 停止训练")
            break

        # 早停 + 保存最佳
        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            best_acc = val_acc
            no_improve = 0
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")
            logger.info(f"  → 保存最佳模型 (val_loss={val_loss:.4f}, acc={val_acc:.3f})")
        else:
            no_improve += 1
            if no_improve >= args.early_stopping_patience:
                logger.warning(f"  ⚠ 早停: {no_improve} epochs 无改善 — 停止训练")
                break

    # ── 保存训练记录 ──
    records = {
        "best_val_loss": best_val_loss,
        "best_val_acc": best_acc,
        "train_loss_log": train_loss_log,
        "val_loss_log": val_loss_log,
        "total_time_seconds": time.time() - t0,
        "epochs_completed": epoch + 1,
        "config": {
            "model": args.model_name,
            "base_adapter": args.base_model,
            "lr": args.lr,
            "beta": args.beta,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_pairs": len(train_dataset),
            "val_pairs": len(val_dataset),
        },
    }
    with open(output_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"训练日志: {output_dir / 'training_log.json'}")
    logger.info(f"最佳模型: {output_dir / 'best_model'} (val_loss={best_val_loss:.4f}, acc={best_acc:.3f})")


if __name__ == "__main__":
    main()
