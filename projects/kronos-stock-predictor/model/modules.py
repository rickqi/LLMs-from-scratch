"""
Kronos 基础模块

包含:
- RMSNorm: Root Mean Square Layer Normalization
- MultiHeadAttention: 多头因果注意力
- TransformerBlock: Transformer 块 (attention + feedforward)
- BinarySphericalQuantizer: 二进制球面量化器 (BSQ)
- HierarchicalEmbedding: 层次化 token 嵌入 (s1 + s2)
- TemporalEmbedding: 时间特征嵌入
- DependencyAwareLayer: 条件依赖层 (s2 以 s1 为条件)
- DualHead: 双头输出 (s1_logits + conditional s2_logits)

参考: Kronos model/module.py (shiyu-coder/Kronos)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama-style)"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# ============================================================
# MultiHeadAttention
# ============================================================

class MultiHeadAttention(nn.Module):
    """多头因果注意力"""

    def __init__(self, d_model: int, n_heads: int, dropout_p: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, D = x.shape

        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        # Padding mask
        if key_padding_mask is not None:
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
            attn_scores = attn_scores.masked_fill(kpm, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


# ============================================================
# TransformerBlock
# ============================================================

class TransformerBlock(nn.Module):
    """Transformer 块: Pre-LN Attention + Pre-LN FFN"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        ffn_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.1,
        resid_dropout_p: float = 0.1,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, attn_dropout_p)
        self.drop1 = nn.Dropout(resid_dropout_p)

        self.ln2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(ffn_dropout_p),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(ffn_dropout_p),
        )
        self.drop2 = nn.Dropout(resid_dropout_p)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Attention sub-layer
        attn_out = self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.drop1(attn_out)

        # FFN sub-layer
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.drop2(ffn_out)

        return x


# ============================================================
# BinarySphericalQuantizer (BSQ)
# 参考: https://arxiv.org/pdf/2406.07548.pdf
# ============================================================

class _BipolarQuantizeSTE(Function):
    """Straight-Through Estimator for bipolar {-1, +1} quantization"""

    @staticmethod
    def forward(ctx, z: torch.Tensor) -> torch.Tensor:
        return torch.where(z > 0, torch.tensor(1.0, device=z.device, dtype=z.dtype),
                           torch.tensor(-1.0, device=z.device, dtype=z.dtype))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def _bipolar_quantize(z: torch.Tensor) -> torch.Tensor:
    """Hard sign forward, identity backward"""
    return _BipolarQuantizeSTE.apply(z)


class BinarySphericalQuantizer(nn.Module):
    """
    Binary Spherical Quantization.

    将连续隐向量 z ∈ R^d 量化为 k-bit 二进制编码 b ∈ {-1, +1}^d
    使用可学习超平面投影 + straight-through estimator。

    Args:
        embed_dim: 总 bit 数 (超平面数量)
        beta: commitment loss 权重
        gamma0: entropy penalty 初始权重
        gamma: entropy penalty 权重
        zeta: 全局 entropy penalty 权重
        group_size: 分组大小 (用于高效熵计算)
    """

    def __init__(
        self,
        embed_dim: int,
        beta: float = 0.25,
        gamma0: float = 1.0,
        gamma: float = 1.0,
        zeta: float = 0.1,
        group_size: int = 9,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.beta = beta
        self.gamma0 = gamma0
        self.gamma = gamma
        self.zeta = zeta

        assert embed_dim % group_size == 0, f"embed_dim ({embed_dim}) must be divisible by group_size ({group_size})"
        self.num_groups = embed_dim // group_size
        self.group_size = group_size

        # 预计算 bit 掩码
        self.register_buffer("basis", 2 ** torch.arange(embed_dim - 1, -1, -1))
        self.register_buffer("group_basis", 2 ** torch.arange(group_size - 1, -1, -1))

        self.num_dimensions = 2 ** embed_dim
        self.bits_per_index = embed_dim

        # Group codebook for entropy estimation
        group_codes = torch.arange(2 ** self.group_size)
        group_codebook = self._indices_to_bits(group_codes).float()[:, -group_size:]
        self.register_buffer("group_codebook", group_codebook, persistent=False)


    def _indices_to_bits(self, indices: torch.Tensor) -> torch.Tensor:
        """将整数索引转为 bits"""
        mask = 2 ** torch.arange(self.embed_dim, device=indices.device, dtype=torch.long)
        bits = (indices.unsqueeze(-1) & mask) != 0
        return bits.float() * 2 - 1  # → {-1, +1}

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Bipolar quantization with STE — operates directly on input."""
        zhat = _bipolar_quantize(z)
        return zhat

    def forward(
        self, z: torch.Tensor, half: bool = False, collect_metrics: bool = True
    ) -> tuple:
        """
        Args:
            z: (B, T, embed_dim) 连续隐向量
            half: 是否只量化前半部分 (用于 encode 时的 half 模式)
            collect_metrics: 是否计算损失 (推理时为 False)

        Returns:
            bsq_loss: 量化损失标量
            quantized: (B, T, embed_dim) 量化后的向量 ({-1, +1} scaled)
            z_indices: 整数索引
        """
        if half:
            s1_dim = self.embed_dim // 2
            z1 = z[:, :, :s1_dim]
            z2 = z[:, :, s1_dim:]

            z1_hat = self.quantize(z1)
            z2_hat = self.quantize(z2)

            quantized_full = torch.cat([z1_hat, z2_hat], dim=-1)
            z_indices = self._bits_to_indices(z1_hat), self._bits_to_indices(z2_hat)
        else:
            quantized_full = self.quantize(z)
            z_indices = self._bits_to_indices(quantized_full)

        # Scale
        q_scale = 1.0 / (self.embed_dim ** 0.5)
        quantized_full = quantized_full * q_scale

        # BSQ loss
        if collect_metrics:
            z_norm = F.normalize(z, dim=-1)
            zhat_norm = F.normalize(quantized_full * (self.embed_dim ** 0.5), dim=-1)
            commit_loss = F.mse_loss(z_norm, zhat_norm.detach()) + self.beta * F.mse_loss(z_norm.detach(), zhat_norm)
            bsq_loss = commit_loss
        else:
            bsq_loss = torch.tensor(0.0, device=z.device)

        # 展开 half 模式的 indices
        if half:
            z_indices = z_indices  # tuple of (s1_indices, s2_indices)

        return bsq_loss, quantized_full, z_indices

    def _bits_to_indices(self, bits: torch.Tensor) -> torch.Tensor:
        """{-1, +1} bits → integer indices. Uses actual last dim size."""
        dim = bits.shape[-1]
        powers = 2 ** torch.arange(dim - 1, -1, -1, device=bits.device, dtype=torch.long)
        binary = (bits > 0).long()
        return (binary * powers).sum(dim=-1)


# ============================================================
# HierarchicalEmbedding
# ============================================================

class HierarchicalEmbedding(nn.Module):
    """
    层次化 token 嵌入: s1 和 s2 各自独立嵌入后相加。

    Args:
        s1_bits: s1 (粗粒度) token 的 bit 数
        s2_bits: s2 (细粒度) token 的 bit 数
        d_model: 嵌入维度
    """

    def __init__(self, s1_bits: int, s2_bits: int, d_model: int):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.d_model = d_model
        self.s1_vocab_size = 2 ** s1_bits
        self.s2_vocab_size = 2 ** s2_bits

        self.emb_s1 = nn.Embedding(self.s1_vocab_size, d_model)
        self.emb_s2 = nn.Embedding(self.s2_vocab_size, d_model)

    def forward(self, token_ids: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            token_ids: [s1_ids (B,T), s2_ids (B,T)]

        Returns:
            (B, T, d_model)
        """
        s1_ids, s2_ids = token_ids
        return self.emb_s1(s1_ids) + self.emb_s2(s2_ids)


# ============================================================
# TemporalEmbedding
# ============================================================

class TemporalEmbedding(nn.Module):
    """
    时间特征嵌入: 将离散时间特征 (minute, hour, weekday, day, month) 嵌入为连续向量后相加。
    """

    def __init__(self, d_model: int, learn_te: bool = True):
        super().__init__()
        self.d_model = d_model
        self.learn_te = learn_te

        if learn_te:
            self.minute_emb = nn.Embedding(60, d_model)
            self.hour_emb = nn.Embedding(24, d_model)
            self.weekday_emb = nn.Embedding(7, d_model)
            self.day_emb = nn.Embedding(32, d_model)
            self.month_emb = nn.Embedding(13, d_model)
        else:
            self.register_buffer("_dummy", torch.zeros(1))

    def forward(self, stamp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stamp: (B, T, 5) — [minute, hour, weekday, day, month] 取整后的值

        Returns:
            (B, T, d_model)
        """
        if not self.learn_te:
            return torch.zeros(*stamp.shape[:2], self.d_model, device=stamp.device)

        minute = self.minute_emb(stamp[:, :, 0].long().clamp(0, 59))
        hour = self.hour_emb(stamp[:, :, 1].long().clamp(0, 23))
        weekday = self.weekday_emb(stamp[:, :, 2].long().clamp(0, 6))
        day = self.day_emb(stamp[:, :, 3].long().clamp(1, 31))
        month = self.month_emb(stamp[:, :, 4].long().clamp(1, 12))

        return minute + hour + weekday + day + month


# ============================================================
# DependencyAwareLayer
# ============================================================

class DependencyAwareLayer(nn.Module):
    """
    条件依赖层: 将 s1 嵌入融入上下文表示，用于 s2 的条件预测。

    x_out = LayerNorm(x + MLP(x + sibling_embed))
    """

    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model),
        )

    def forward(
        self, x: torch.Tensor, sibling_embed: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused = self.norm(x + sibling_embed)
        delta = self.fuse(fused)
        return x + delta


# ============================================================
# DualHead
# ============================================================

class DualHead(nn.Module):
    """
    双头输出:
    - head: s1 预测 (Linear(d_model, s1_vocab))
    - cond_head: s2 条件预测 (Linear(d_model, s2_vocab))
    """

    def __init__(self, s1_bits: int, s2_bits: int, d_model: int):
        super().__init__()
        self.s1_vocab_size = 2 ** s1_bits
        self.s2_vocab_size = 2 ** s2_bits

        self.head = nn.Linear(d_model, self.s1_vocab_size)
        self.cond_head = nn.Linear(d_model, self.s2_vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """默认 forward → s1 logits"""
        return self.head(x)

    def cond_forward(self, x: torch.Tensor) -> torch.Tensor:
        """条件 forward → s2 logits"""
        return self.cond_head(x)
