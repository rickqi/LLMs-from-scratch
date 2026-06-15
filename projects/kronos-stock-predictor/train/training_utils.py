"""
训练工具函数 — DDP 设置、随机种子、日志
"""

import os
import random
import time
import logging

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def set_seed(seed: int = 100):
    """设置全局随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_ddp(rank: int, world_size: int):
    """初始化分布式训练环境"""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def get_model_size(model: torch.nn.Module) -> tuple[int, str]:
    """返回模型参数量和可读字符串"""
    num_params = sum(p.numel() for p in model.parameters())
    if num_params >= 1e9:
        return num_params, f"{num_params / 1e9:.1f}B"
    elif num_params >= 1e6:
        return num_params, f"{num_params / 1e6:.1f}M"
    else:
        return num_params, f"{num_params / 1e3:.1f}K"


def format_time(seconds: float) -> str:
    """将秒数格式化为可读字符串"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m {seconds % 60:.0f}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def get_device() -> torch.device:
    """自动检测可用设备"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
):
    """保存训练 checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    filepath: str,
) -> tuple[int, float]:
    """加载训练 checkpoint"""
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(f"Checkpoint loaded: {filepath} (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"], checkpoint["loss"]
