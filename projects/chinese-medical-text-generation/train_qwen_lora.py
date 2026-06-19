"""
Qwen3 + LoRA 微调: 中文医学诊疗指南文本生成
===========================================
用法:
  # 纯续写微调
  python train_qwen_lora.py --data_dir ./data_full --output_dir ./output_full --epochs 5
  # 指令微调 (从续写模型续训)
  python train_qwen_lora.py --resume_from ./output_full/best_model \\
      --instruction_data ./docs/med_instruction_train_chatml.json \\
      --instruction_val_data ./docs/med_instruction_val_chatml.json \\
      --output_dir ./output_inst --epochs 1 --lr 1e-5 --instruction_ratio 0.4

训练标准 (内置):
  - 早停: patience 步无改善 (min_delta 阈值) → 自动停止
  - 过拟合检测: train/val gap > 阈值 → 自动停止
  - 指令验证集: 独立 hold-out QA 评估 (非续写数据)
  - 仅保留 best_model (不保存退化的 final_model)

依赖:
  pip install transformers peft datasets accelerate
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import time
import random
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
EARLY_STOPPING_PATIENCE = 50
MIN_DELTA = 0.001
OVERFIT_GAP_THRESHOLD = 0.5


class MedicalTextDataset(Dataset):
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


class InstructionDataset(Dataset):
    """加载 ChatML 格式的指令微调数据，仅对 assistant 部分计算 loss"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 768):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items = raw.get("data", raw if isinstance(raw, list) else [])
        self.examples = []
        for item in items:
            messages = item["messages"]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            tokenized = tokenizer(text, truncation=True, max_length=max_length, return_tensors=None)
            input_ids = tokenized["input_ids"]
            labels = input_ids.copy()
            # mask all tokens except assistant response
            assist_start = self._find_assistant_start(input_ids)
            for i in range(assist_start):
                labels[i] = -100
            self.examples.append({"input_ids": input_ids, "labels": labels})

    def _find_assistant_start(self, input_ids):
        assist_marker = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        for i in range(len(input_ids) - len(assist_marker) + 1):
            if input_ids[i:i+len(assist_marker)] == assist_marker:
                return i + len(assist_marker)
        return 0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MixedDataset(Dataset):
    """混合指令数据 (80%) 和纯续写数据 (20%)，防灾难性遗忘"""

    def __init__(self, inst_dataset, cont_dataset, inst_ratio=0.8):
        self.inst = inst_dataset
        self.cont = cont_dataset
        self.inst_ratio = inst_ratio
        self.total = len(inst_dataset) + len(cont_dataset)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if random.random() < self.inst_ratio:
            return self.inst[random.randint(0, len(self.inst) - 1)]
        else:
            return self.cont[random.randint(0, len(self.cont) - 1)]


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
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="基座模型名称")
    parser.add_argument("--resume_from", type=str, default=None, help="已有 LoRA adapter 路径 (续训)")
    parser.add_argument("--instruction_data", type=str, default=None, help="ChatML 指令微调数据路径")
    parser.add_argument("--instruction_val_data", type=str, default=None, help="ChatML 指令验证集路径 (不提供则从 instruction_data 自动切 50 条)")
    parser.add_argument("--instruction_ratio", type=float, default=0.4, help="指令数据占比 (默认0.4)")
    parser.add_argument("--early_stopping_patience", type=int, default=EARLY_STOPPING_PATIENCE, help="早停耐心步数 (默认50)")
    parser.add_argument("--min_delta", type=float, default=MIN_DELTA, help="val_loss 改善最小阈值 (默认0.001)")
    parser.add_argument("--overfit_gap_threshold", type=float, default=OVERFIT_GAP_THRESHOLD, help="train/val gap 过拟合阈值 (默认0.5)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info(f"使用设备: {device}")

    # 1. 加载 tokenizer 和模型
    logger.info(f"加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    model = model.to(device)

    # 2. 配置 LoRA (支持 resume)
    from peft import PeftModel
    if args.resume_from:
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
        adapter_config = json.load(open(Path(args.resume_from) / "adapter_config.json"))
        logger.info(f"从已有 LoRA 权重续训: {args.resume_from} (rank={adapter_config['r']})")
    else:
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
    tokenizer.model_max_length = args.max_length
    cont_train = MedicalTextDataset(str(data_dir / "train.txt"), tokenizer, args.max_length)
    inst_val_dataset = None
    inst_train_items = 0

    if args.instruction_data:
        logger.info(f"加载指令数据: {args.instruction_data}")
        inst_dataset = InstructionDataset(args.instruction_data, tokenizer, args.max_length)
        inst_train_items = len(inst_dataset)
        train_dataset = MixedDataset(inst_dataset, cont_train, args.instruction_ratio)

        # 指令验证集: 优先用独立文件, 否则自动从训练集切分
        val_data_path = args.instruction_val_data
        if val_data_path and Path(val_data_path).exists():
            logger.info(f"加载指令验证集: {val_data_path}")
            inst_val_dataset = InstructionDataset(val_data_path, tokenizer, args.max_length)
            logger.info(f"指令验证样本: {len(inst_val_dataset)} (独立 hold-out)")
        else:
            logger.warning("未提供 --instruction_val_data, 自动从指令数据切分 50 条")
            import random as _random
            _random.seed(42)
            indices = list(range(len(inst_dataset)))
            _random.shuffle(indices)
            val_indices = set(indices[:50])
            inst_val_dataset = InstructionDataset.__new__(InstructionDataset)
            inst_val_dataset.tokenizer = tokenizer
            inst_val_dataset.max_length = args.max_length
            inst_val_dataset.examples = [inst_dataset[i] for i in val_indices]
            # 从训练集剔除验证集样本
            inst_dataset.examples = [inst_dataset[i] for i in range(len(inst_dataset)) if i not in val_indices]
            train_dataset = MixedDataset(inst_dataset, cont_train, args.instruction_ratio)
            logger.info(f"指令验证样本: {len(inst_val_dataset)} (自动切分)")

        logger.info(f"指令训练: {len(inst_dataset)}, 续写训练: {len(cont_train)}, 混合比: {args.instruction_ratio}")
    else:
        train_dataset = cont_train
        logger.info(f"加载数据: {data_dir}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # 验证集: 指令模式用指令验证集, 纯续写模式用续写验证集
    if inst_val_dataset is not None:
        val_loader = DataLoader(
            inst_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    else:
        val_dataset = MedicalTextDataset(str(data_dir / "val.txt"), tokenizer, args.max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    logger.info(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_loader.dataset)}")
    logger.info(f"训练批次/epoch: {len(train_loader)}, 验证批次: {len(val_loader)}")

    # 4. 优化器 & 调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )

    # 5. 断点恢复
    checkpoint_path = output_dir / "checkpoint"
    global_step = 0
    best_val_loss = float("inf")
    train_loss_log = []
    val_loss_log = []
    start_epoch = 1
    elapsed_offset = 0.0

    if checkpoint_path.exists():
        logger.info(f"发现断点检查点: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path / "training_state.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        train_loss_log = ckpt.get("train_loss_log", [])
        val_loss_log = ckpt.get("val_loss_log", [])
        start_epoch = ckpt.get("epoch", 1)
        elapsed_offset = ckpt.get("elapsed", 0.0)
        logger.info(f"从 epoch {start_epoch}, step {global_step} 恢复训练 (best_val={best_val_loss:.4f})")

    # 6. 训练循环 (含早停 + 过拟合检测)
    logger.info(f"开始训练... (早停patience={args.early_stopping_patience}, min_delta={args.min_delta}, "
                f"overfit_gap={args.overfit_gap_threshold})")
    start_time = time.time()
    steps_no_improvement = 0
    stopped_early = False
    stopped_reason = ""

    for epoch in range(start_epoch, args.epochs + 1):
        if stopped_early:
            break
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

                    elapsed = elapsed_offset + (time.time() - start_time)
                    gap = avg_train_loss - val_loss
                    logger.info(
                        f"Epoch {epoch}/{args.epochs} | "
                        f"Step {global_step:06d} | "
                        f"Train loss: {avg_train_loss:.4f} | "
                        f"Val loss: {val_loss:.4f} | "
                        f"Gap: {gap:+.4f} | "
                        f"Patience: {steps_no_improvement}/{args.early_stopping_patience} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"Time: {elapsed:.0f}s"
                    )

                    # ─── 过拟合检测 ───
                    if gap > args.overfit_gap_threshold:
                        stopped_early = True
                        stopped_reason = f"过拟合 (train/val gap={gap:.4f} > {args.overfit_gap_threshold})"
                        logger.warning(f"  ⚠ {stopped_reason} — 停止训练")
                        break

                    # ─── 最佳模型保存 + 早停计数 ───
                    if val_loss < best_val_loss - args.min_delta:
                        best_val_loss = val_loss
                        steps_no_improvement = 0
                        model.save_pretrained(output_dir / "best_model")
                        tokenizer.save_pretrained(output_dir / "best_model")
                        logger.info(f"  → 保存最佳模型 (val_loss={val_loss:.4f})")
                    else:
                        steps_no_improvement += 10
                        if steps_no_improvement >= args.early_stopping_patience:
                            stopped_early = True
                            stopped_reason = (
                                f"早停 ({steps_no_improvement} 步无改善, "
                                f"best_val={best_val_loss:.4f}, min_delta={args.min_delta})"
                            )
                            logger.warning(f"  ⚠ {stopped_reason} — 停止训练")
                            break

        # 每 epoch 保存断点检查点
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        elapsed = elapsed_offset + (time.time() - start_time)
        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss_log": train_loss_log,
            "val_loss_log": val_loss_log,
            "elapsed": elapsed,
        }, checkpoint_path / "training_state.pt")
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        logger.info(f"  → 断点已保存 (epoch {epoch}, step {global_step})")

        if stopped_early:
            break

        # 每个 epoch 生成样本
        if args.instruction_data:
            inst_prompts = [
                "胃癌的典型临床表现有哪些？",
                "肺癌的TNM分期标准是什么？",
                "手术后需要观察哪些并发症？",
                "请对比CT和MRI在肿瘤分期中的优缺点。",
            ]
            for q in inst_prompts:
                msg = [{"role": "user", "content": q}]
                prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                sample = generate_sample(model, tokenizer, prompt, device, max_new_tokens=200)
                logger.info(f"\n[指令生成 | Q: {q}]\n{sample}\n")
        else:
            for prompt in ["临床表现：", "诊断依据：", "治疗方案：", "预后判断："]:
                sample = generate_sample(model, tokenizer, prompt, device)
                logger.info(f"\n[生成样本 | prompt: {prompt}]\n{sample}\n")

    if stopped_early:
        logger.info(f"训练提前终止: {stopped_reason}")

    # 7. 保存训练记录 (仅 best_model, 不保存退化的 final_model)
    records = {
        "train_loss": train_loss_log,
        "val_loss": val_loss_log,
        "best_val_loss": best_val_loss,
        "total_time_seconds": elapsed_offset + (time.time() - start_time),
        "global_step": global_step,
        "config": {
            "model": args.model_name,
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
