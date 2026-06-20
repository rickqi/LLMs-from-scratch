"""
Kronos Tokenizer 训练脚本

训练 Stage 1: K-line Tokenizer (Transformer 自编码器 + BSQ 量化)

用法:
    # 单 GPU
    python train/train_tokenizer.py --data_dir data/processed --output_dir outputs/tokenizer --model_size mini

    # 多 GPU (DDP)
    torchrun --standalone --nproc_per_node=NUM_GPUS train/train_tokenizer.py --data_dir data/processed ...
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.default_config import Config
from config.model_configs import build_tokenizer_config, get_model_config
from data.dataset import StockDataset
from model.kronos_tokenizer import KronosTokenizer
from train.losses import tokenizer_loss
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
    """创建训练和验证 DataLoader"""
    train_dataset = StockDataset(config.dataset_path, config, data_type="train")
    val_dataset = StockDataset(config.dataset_path, config, data_type="val")

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def train_epoch(
    model: KronosTokenizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config: Config,
    epoch: int,
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (batch_x, batch_x_stamp) in enumerate(dataloader):
        batch_x = batch_x.to(device, non_blocking=True)

        optimizer.zero_grad()

        (z_pre, z_full), bsq_loss, _, _ = model(batch_x)

        loss = tokenizer_loss(batch_x, z_pre, z_full, bsq_loss)

        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % config.log_interval == 0:
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}" if scheduler else ""
            )

    return total_loss / n_batches


@torch.no_grad()
def validate(model: KronosTokenizer, dataloader: DataLoader, device: torch.device) -> float:
    """验证"""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch_x, batch_x_stamp in dataloader:
        batch_x = batch_x.to(device, non_blocking=True)
        (z_pre, z_full), bsq_loss, _, _ = model(batch_x)
        loss = tokenizer_loss(batch_x, z_pre, z_full, bsq_loss)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train Kronos Tokenizer")
    parser.add_argument("--data_dir", type=str, default="./data/processed", help="预处理数据目录")
    parser.add_argument("--output_dir", type=str, default="./outputs/tokenizer", help="模型输出目录")
    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "small", "base"])
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=50, help="批大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--lookback", type=int, default=None, help="覆盖 lookback_window")
    parser.add_argument("--feature_dim", type=int, default=None, help="覆盖输入特征维度")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="早停耐心 epochs")
    parser.add_argument("--min_delta", type=float, default=0.001, help="val improvement min threshold")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda:0 / cpu)")
    args = parser.parse_args()

    # 配置
    config = Config()
    config.dataset_path = args.data_dir
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.tokenizer_learning_rate = args.lr
    if args.lookback is not None:
        config.lookback_window = args.lookback

    set_seed(config.seed)

    device = torch.device(args.device) if args.device else get_device()
    logger.info(f"Using device: {device}")

    # 创建模型
    tokenizer_cfg = build_tokenizer_config(args.model_size)
    model = KronosTokenizer(**tokenizer_cfg)
    model.to(device)

    n_params, n_params_str = get_model_size(model)
    logger.info(f"Tokenizer params: {n_params_str} ({n_params:,})")

    # 数据
    train_loader, val_loader = create_dataloaders(config)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.tokenizer_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
    )
    # Cosine LR with linear warmup
    total_steps = config.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    elapsed_offset = 0.0

    # 断点恢复
    checkpoint_dir = f"{args.output_dir}/checkpoint"
    ckpt_file = f"{checkpoint_dir}/training_state.pt"
    if os.path.exists(ckpt_file):
        logger.info(f"发现断点: {ckpt_file}")
        state = load_checkpoint(model, optimizer, scheduler, ckpt_file)
        start_epoch = state["epoch"]
        best_val_loss = state["loss"]
        train_losses = state.get("train_losses", [])
        val_losses = state.get("val_losses", [])
        elapsed_offset = state.get("elapsed", 0.0)
        logger.info(f"从 epoch {start_epoch} 恢复 (best_val={best_val_loss:.4f})")

    # 训练循环
    os.makedirs(checkpoint_dir, exist_ok=True)
    t_start = time.time()
    steps_no_improvement = 0
    stopped_early = False

    for epoch in range(start_epoch, config.epochs):
        if stopped_early:
            break
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, config, epoch)
        val_loss = validate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Time: {format_time(epoch_time)}"
        )

        # 保存最佳模型
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            steps_no_improvement = 0
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss,
                            train_losses, val_losses, elapsed_offset + (time.time() - t_start),
                            f"{args.output_dir}/best_model.pt")
            logger.info(f"  ✓ New best model (val_loss={val_loss:.4f})")
        else:
            steps_no_improvement += 1
            logger.info(f"  No improvement ({steps_no_improvement}/{args.early_stopping_patience})")

        if steps_no_improvement >= args.early_stopping_patience:
            stopped_early = True
            logger.warning(f"  ⚠ Early stopping at epoch {epoch+1}")
            break

        # 每个 epoch 保存完整训练状态（支持断点续传）
        save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss,
                        train_losses, val_losses, elapsed_offset + (time.time() - t_start),
                        ckpt_file)
        logger.info(f"  -> 断点已保存 (epoch {epoch+1})")

    total_time = time.time() - t_start
    if stopped_early:
        logger.info(f"Training stopped early at epoch {start_epoch + steps_no_improvement}")
    logger.info(f"Training complete. Total time: {format_time(total_time)}")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {args.output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
