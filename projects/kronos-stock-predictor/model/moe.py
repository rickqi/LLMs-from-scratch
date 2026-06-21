"""
MoE (Mixture of Experts) Layer for Kronos Transformer

Sparse MoE with top-2 routing, replacing the standard FFN in TransformerBlock.
Reference: FinCast (CIKM 2025), Switch Transformer (Fedus et al. 2022)

Usage:
    from model.moe import MoELayer
    
    # Replace FFN in TransformerBlock with MoE
    moe = MoELayer(d_model=64, ff_dim=256, num_experts=4, top_k=2)
    x = moe(x)  # (B, T, d_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MoELayer(nn.Module):
    """Sparse Mixture of Experts FFN layer.
    
    Args:
        d_model: model dimension
        ff_dim: feed-forward dimension per expert
        num_experts: total number of experts
        top_k: number of experts to activate per token (default 2)
        capacity_factor: expert capacity = (tokens/num_experts) * capacity_factor
        dropout: dropout rate in expert FFN
    """
    
    def __init__(
        self,
        d_model: int = 64,
        ff_dim: int = 256,
        num_experts: int = 4,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # Router: linear projection to expert logits
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # Expert FFNs: each is a standard SwiGLU MLP
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, ff_dim, dropout) for _ in range(num_experts)
        ])
        
        # For tracking load balancing
        self.register_buffer('expert_counts', torch.zeros(num_experts))
    
    def forward(self, x: torch.Tensor, return_aux_loss: bool = False):
        """Forward pass with top-k sparse routing.
        
        Args:
            x: (B, T, d_model) or (B, d_model)
            return_aux_loss: if True, also returns load balancing loss
            
        Returns:
            output: same shape as x
            aux_loss (optional): load balancing loss
        """
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.reshape(-1, D)  # (B*T, D)
        else:
            x_flat = x
            B, T = 1, x.shape[0]
        
        # Router scores
        router_logits = self.router(x_flat)  # (N, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # re-normalize
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Track load balancing
        if self.training:
            self.expert_counts.zero_()
        
        # Dispatch to experts (simple loop for clarity — batch dispatch for speed)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]  # (N,)
            weight = top_k_weights[:, k].unsqueeze(-1)  # (N, 1)
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_out = self.experts[e](x_flat[mask])
                    output[mask] += weight[mask] * expert_out
                    if self.training:
                        self.expert_counts[e] += mask.sum().float()
        
        # Reshape back
        if len(orig_shape) == 3:
            output = output.reshape(B, T, D)
        
        if return_aux_loss:
            # Load balancing loss: encourage uniform expert usage
            # Compute fraction of tokens dispatched to each expert
            N = x_flat.shape[0]
            if N > 0:
                expert_fraction = self.expert_counts / N
                # Target: uniform distribution
                target_fraction = torch.ones_like(expert_fraction) / self.num_experts
                # Mean router probability per expert
                mean_router_prob = router_probs.mean(dim=0)
                # Aux loss = num_experts * sum(f_i * P_i)
                aux_loss = self.num_experts * (expert_fraction * mean_router_prob).sum()
            else:
                aux_loss = torch.tensor(0.0, device=x.device)
            return output, aux_loss
        
        return output
    
    def get_expert_usage(self) -> dict:
        """Get expert utilization statistics"""
        total = self.expert_counts.sum().item()
        if total == 0:
            return {f'expert_{i}': 0.0 for i in range(self.num_experts)}
        return {f'expert_{i}': (self.expert_counts[i].item() / total * 100)
                for i in range(self.num_experts)}


class ExpertFFN(nn.Module):
    """Single expert FFN with SwiGLU activation (matches TransformerBlock FFN style)"""
    
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim)
        self.w2 = nn.Linear(d_model, ff_dim)  # gate
        self.w3 = nn.Linear(ff_dim, d_model)  # output
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE replacing the standard FFN.
    
    Compatible drop-in replacement for TransformerBlock with MoE support.
    """
    
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        ff_dim: int = 256,
        ffn_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.1,
        resid_dropout_p: float = 0.1,
        use_moe: bool = True,
        num_experts: int = 4,
        moe_top_k: int = 2,
    ):
        super().__init__()
        self.use_moe = use_moe
        
        # Attention
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout_p, batch_first=True
        )
        self.dropout1 = nn.Dropout(resid_dropout_p)
        
        # FFN or MoE
        self.ln2 = nn.RMSNorm(d_model)
        if use_moe:
            self.moe = MoELayer(
                d_model=d_model, ff_dim=ff_dim, num_experts=num_experts,
                top_k=moe_top_k, dropout=ffn_dropout_p
            )
            self.ffn = None
        else:
            self.moe = None
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ff_dim),
                nn.SiLU(),
                nn.Dropout(ffn_dropout_p),
                nn.Linear(ff_dim, d_model),
                nn.Dropout(ffn_dropout_p),
            )
        self.dropout2 = nn.Dropout(resid_dropout_p)
    
    def forward(self, x: torch.Tensor, key_padding_mask=None, return_aux_loss: bool = False):
        # Self-attention
        residual = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.dropout1(x)
        x = x + residual
        
        # FFN or MoE
        residual = x
        x = self.ln2(x)
        aux_loss = None
        if self.use_moe and self.moe is not None:
            if return_aux_loss:
                x, aux_loss = self.moe(x, return_aux_loss=True)
            else:
                x = self.moe(x)
        else:
            x = self.ffn(x)
        x = self.dropout2(x)
        x = x + residual
        
        if return_aux_loss:
            return x, aux_loss
        return x
