"""
Kronos RL Trainer — GRPO + PPO for Sharpe Ratio optimization

Supports:
  GRPO (Group Relative Policy Optimization): sample-based relative advantage
  PPO  (Proximal Policy Optimization): clipped surrogate + KL penalty

Usage:
    python scripts/train_kronos_rl.py \
        --predictor_path ./outputs/predictor/best_model.pt \
        --data_dir ./data/processed_real \
        --method grpo --epochs 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_sharpe(returns: np.ndarray, top_k: int = 20) -> float:
    """Top-K stock selection Sharpe Ratio reward"""
    if len(returns) < top_k:
        return 0.0
    idx = np.argsort(returns)[-top_k:]
    selected = returns[idx]
    mean_ret = np.mean(selected)
    std_ret = np.std(selected) + 1e-8
    return mean_ret / std_ret


def compute_rankic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """RankIC reward (Spearman correlation)"""
    from scipy.stats import spearmanr
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    if mask.sum() < 10:
        return 0.0
    ic, _ = spearmanr(predictions[mask], actuals[mask])
    return float(ic) if not np.isnan(ic) else 0.0


class KronosRL:
    """RL trainer for Kronos models with GRPO and PPO support.
    
    Uses Sharpe Ratio as the primary reward signal.
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        lr: float = 1e-4,
        beta: float = 0.1,       # KL penalty coefficient
        clip_epsilon: float = 0.2, # PPO clip range
        group_size: int = 8,      # GRPO group samples
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device) if ref_model is not None else None
        self.device = device
        self.beta = beta
        self.clip_epsilon = clip_epsilon
        self.group_size = group_size
        
        # Optimizer (only trainable params)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
    
    # ─────────────────────────────────────────────────────────────
    # GRPO: Group Relative Policy Optimization
    # ─────────────────────────────────────────────────────────────
    
    def grpo_step(
        self,
        features: torch.Tensor,   # (B, T, d_in)
        returns: torch.Tensor,    # (B, T) actual returns
        top_k: int = 20,
    ) -> dict:
        """Single GRPO optimization step.
        
        1. Sample group_size predictions per stock
        2. Compute Sharpe for each sample
        3. Use relative advantage = individual - mean Sharpe
        4. Update via policy gradient
        """
        B = features.shape[0]
        self.model.train()
        
        all_losses = []
        all_sharpes = []
        
        # Generate group of predictions
        for g in range(self.group_size):
            with torch.no_grad():
                pred = self.model.sample_prediction(features).detach()
            
            # Compute reward (Sharpe)
            sharpe = compute_sharpe(pred.cpu().numpy().flatten(), top_k)
            all_sharpes.append(sharpe)
        
        mean_sharpe = np.mean(all_sharpes)
        advantages = [s - mean_sharpe for s in all_sharpes]
        
        # Policy gradient update
        total_loss = 0.0
        for g in range(self.group_size):
            self.optimizer.zero_grad()
            
            pred = self.model.sample_prediction(features)
            log_prob = self.model.log_prob(features, pred)
            
            # Policy loss = -advantage * log_prob
            loss = -advantages[g] * log_prob.mean()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return {
            'loss': total_loss / self.group_size,
            'mean_sharpe': mean_sharpe,
            'max_sharpe': max(all_sharpes),
            'min_sharpe': min(all_sharpes),
        }
    
    # ─────────────────────────────────────────────────────────────
    # PPO: Proximal Policy Optimization
    # ─────────────────────────────────────────────────────────────
    
    def ppo_step(
        self,
        features: torch.Tensor,
        returns: torch.Tensor,
        old_predictions: Optional[torch.Tensor] = None,
        top_k: int = 20,
    ) -> dict:
        """Single PPO optimization step with clipped surrogate objective."""
        self.model.train()
        
        # Get old policy predictions (for importance sampling)
        with torch.no_grad():
            if old_predictions is None:
                old_predictions = self.model.sample_prediction(features).detach()
            old_log_prob = self.model.log_prob(features, old_predictions).detach()
        
        # Compute advantage (Sharpe-based)
        with torch.no_grad():
            sharpe = compute_sharpe(returns.cpu().numpy().flatten(), top_k)
        
        self.optimizer.zero_grad()
        
        # New policy
        new_pred = self.model.sample_prediction(features)
        new_log_prob = self.model.log_prob(features, new_pred)
        
        # Importance sampling ratio
        ratio = torch.exp(new_log_prob - old_log_prob)
        
        # Clipped surrogate objective
        surr1 = ratio * sharpe
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * sharpe
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL penalty (prevent divergence from reference)
        kl_loss = 0.0
        if self.ref_model is not None:
            with torch.no_grad():
                ref_pred = self.ref_model(features).detach()
            kl = self._compute_kl(new_pred, ref_pred)
            kl_loss = self.beta * kl
        
        loss = policy_loss + kl_loss
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_loss': float(kl_loss) if isinstance(kl_loss, torch.Tensor) else kl_loss,
            'sharpe': sharpe,
            'ratio_mean': ratio.mean().item(),
        }
    
    def _compute_kl(self, pred: torch.Tensor, ref_pred: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, ref_pred)
