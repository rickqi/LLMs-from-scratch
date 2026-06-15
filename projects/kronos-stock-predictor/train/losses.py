"""
训练损失函数 — Kronos 层次化预测损失
"""

import torch
import torch.nn.functional as F


def tokenizer_loss(
    x: torch.Tensor,
    x_recon_coarse: torch.Tensor,
    x_recon_fine: torch.Tensor,
    bsq_loss: torch.Tensor,
    lambda_quant: float = 1.0,
) -> torch.Tensor:
    """
    Tokenizer 层次化重建损失。

    L = MSE(x_recon_coarse, x) + MSE(x_recon_fine, x) + λ * BSQ_loss

    设计意图:
    - L_coarse: 粗 token (s1) 学习主导价格走势
    - L_fine: 完整 token 重建 — 细 token (s2) 学习残差细节
    - BSQ_loss: 量化正则化 (commitment + entropy)
    """
    l_coarse = F.mse_loss(x_recon_coarse, x)
    l_fine = F.mse_loss(x_recon_fine, x)
    return l_coarse + l_fine + lambda_quant * bsq_loss


def predictor_loss(
    s1_logits: torch.Tensor,
    s2_logits: torch.Tensor,
    s1_targets: torch.Tensor,
    s2_targets: torch.Tensor,
    s1_weight: float = 1.0,
    s2_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predictor 层次化交叉熵损失。

    L = w1 * CE(s1_logits, s1_targets) + w2 * CE(s2_logits, s2_targets)

    Args:
        s1_logits: (B, T, vocab_s1) — s1 token 预测 logits
        s2_logits: (B, T, vocab_s2) — s2 token 预测 logits
        s1_targets: (B, T) — s1 token 真实标签
        s2_targets: (B, T) — s2 token 真实标签
        s1_weight: s1 损失权重
        s2_weight: s2 损失权重

    Returns:
        total_loss, loss_s1, loss_s2
    """
    vocab_s1 = s1_logits.size(-1)
    vocab_s2 = s2_logits.size(-1)

    loss_s1 = F.cross_entropy(
        s1_logits.reshape(-1, vocab_s1),
        s1_targets.reshape(-1),
    )
    loss_s2 = F.cross_entropy(
        s2_logits.reshape(-1, vocab_s2),
        s2_targets.reshape(-1),
    )

    total_loss = s1_weight * loss_s1 + s2_weight * loss_s2
    return total_loss, loss_s1, loss_s2
