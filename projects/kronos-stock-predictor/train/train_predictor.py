"""
Kronos Predictor 训练脚本

训练 Stage 2: 自回归预测模型 (Decoder-only Transformer)

用法:
    # 单 GPU
    python train/train_predictor.py --tokenizer_path outputs/tokenizer/best_model.pt --data_dir data/processed --model_size mini

    # 多 GPU (DDP)
    torchrun --standalone --nproc_per_node=NUM_GPUS train/train_predictor.py ...
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.default_config import Config
from config.model_configs import build_model_config, get_model_config
from data.dataset import StockDataset
from model.kronos_tokenizer import KronosTokenizer
from model.kronos_model import Kronos
from train.losses import predictor_loss
from train.training_utils import (
    set_seed,
    get_device,
    get_model_size,
    format_time,
    save_checkpoint,
    load_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    train_dataset = StockDataset(config.dataset_path, config, data_type="train")
    val_dataset = StockDataset(config.dataset_path, config, data_type="val")
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader


def train_epoch(
    model: Kronos,
    tokenizer: KronosTokenizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config: Config,
    epoch: int,
) -> float:
    """训练一个 epoch"""
    model.train()
    tokenizer.eval()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (batch_x, batch_x_stamp) in enumerate(dataloader):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

        # Tokenize on-the-fly (no grad for tokenizer)
        with torch.no_grad():
            token_indices = tokenizer.encode(batch_x, half=True)
            s1_ids = token_indices[0]  # (B, T)
            s2_ids = token_indices[1]  # (B, T)

        # Teacher forcing: predict next token
        # Input:  tokens[:, :-1]
        # Target: tokens[:, 1:]
        input_s1 = s1_ids[:, :-1]
        input_s2 = s2_ids[:, :-1]
        target_s1 = s1_ids[:, 1:]
        target_s2 = s2_ids[:, 1:]

        # Time features: use all but last for input
        input_stamp = batch_x_stamp[:, :-1, :]

        optimizer.zero_grad()

        s1_logits, s2_logits = model(
            input_s1, input_s2, stamp=input_stamp,
            use_teacher_forcing=False,
        )

        loss, loss_s1, loss_s2 = predictor_loss(
            s1_logits, s2_logits, target_s1, target_s2,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % config.log_interval == 0:
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} (s1: {loss_s1.item():.4f}, s2: {loss_s2.item():.4f})"
            )

    return total_loss / n_batches


@torch.no_grad()
def validate(
    model: Kronos,
    tokenizer: KronosTokenizer,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    tokenizer.eval()
    total_loss = 0.0
    n_batches = 0

    for batch_x, batch_x_stamp in dataloader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

        token_indices = tokenizer.encode(batch_x, half=True)
        s1_ids = token_indices[0]
        s2_ids = token_indices[1]

        input_s1 = s1_ids[:, :-1]
        input_s2 = s2_ids[:, :-1]
        target_s1 = s1_ids[:, 1:]
        target_s2 = s2_ids[:, 1:]
        input_stamp = batch_x_stamp[:, :-1, :]

        s1_logits, s2_logits = model(input_s1, input_s2, stamp=input_stamp)

        loss, _, _ = predictor_loss(s1_logits, s2_logits, target_s1, target_s2)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train Kronos Predictor")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="预训练 Tokenizer checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--output_dir", type=str, default="./outputs/predictor")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "small", "base"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    config.dataset_path = args.data_dir
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.predictor_learning_rate = args.lr
    if args.lookback is not None:
        config.lookback_window = args.lookback

    set_seed(config.seed)
    device = torch.device(args.device) if args.device else get_device()
    logger.info(f"Using device: {device}")

    # 加载 Tokenizer（冻结）
    mc = get_model_config(args.model_size)
    from config.model_configs import build_tokenizer_config
    tokenizer_cfg = build_tokenizer_config(args.model_size)
    tokenizer = KronosTokenizer(**tokenizer_cfg)
    tokenizer.to(device)

    if os.path.exists(args.tokenizer_path):
        ckpt = torch.load(args.tokenizer_path, map_location=device)
        tokenizer.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Tokenizer loaded from: {args.tokenizer_path}")
    else:
        logger.warning(f"Tokenizer path not found: {args.tokenizer_path}, using random init")

    for param in tokenizer.parameters():
        param.requires_grad = False

    # 创建 Predictor
    model_cfg = build_model_config(args.model_size)
    model = Kronos(**model_cfg)
    model.to(device)

    n_params, n_params_str = get_model_size(model)
    logger.info(f"Predictor params: {n_params_str} ({n_params:,})")

    # 数据
    train_loader, val_loader = create_dataloaders(config)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.predictor_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.predictor_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.epochs,
        pct_start=0.03,
        div_factor=10,
    )

    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume)

    best_val_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    t_start = time.time()
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()

        train_loss = train_epoch(model, tokenizer, train_loader, optimizer, scheduler, device, config, epoch)
        val_loss = validate(model, tokenizer, val_loader, device)

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Time: {format_time(epoch_time)}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch + 1, val_loss, f"{args.output_dir}/best_model.pt")
            logger.info(f"  ✓ New best model (val_loss={val_loss:.4f})")

        save_checkpoint(model, optimizer, epoch + 1, val_loss, f"{args.output_dir}/checkpoint_epoch{epoch+1}.pt")

    total_time = time.time() - t_start
    logger.info(f"Training complete. Total time: {format_time(total_time)}")
    logger.info(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
