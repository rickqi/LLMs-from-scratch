"""
KronosTokenizer — Stage 1 of the Kronos two-stage framework.

A Transformer autoencoder that quantizes continuous OHLCV data into
hierarchical discrete tokens.  The first half of the codebook bits (s1)
captures coarse / dominant price trends, while the second half (s2)
captures fine-grained residual details.

Architecture overview::

    Input (B, T, d_in)
      │
      ▼  embed (Linear d_in → d_model)
      │
      ▼  encoder (n_enc_layers-1 × TransformerBlock)
      │
      ▼  quant_embed (Linear d_model → s1_bits+s2_bits)
      │
      ▼  BinarySphericalQuantizer → (bsq_loss, quantized, z_indices)
      │
      ├─► quantized[:,:,:s1_bits] ─► post_quant_embed_pre ─► decoder ─► head ─► z_pre_recon
      │
      └─► quantized ─────────────► post_quant_embed ─────► decoder ─► head ─► z_full_recon

References
----------
Kronos: A Learned Tokenizer for Financial Time-Series Forecasting.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from model.modules import TransformerBlock, BinarySphericalQuantizer


class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    """Transformer autoencoder that quantizes OHLCV sequences into
    hierarchical discrete tokens (s1 coarse + s2 fine).

    Parameters
    ----------
    d_in : int
        Input feature dimension (default 6 for OHLCVA).
    d_model : int
        Internal Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    ff_dim : int
        Feed-forward intermediate dimension.
    n_enc_layers : int
        Total encoder layers (first is the embed projection, rest are TransformerBlocks).
    n_dec_layers : int
        Total decoder layers (first is the post-quant projection, rest are TransformerBlocks).
    ffn_dropout_p : float
        Dropout probability for feed-forward layers.
    attn_dropout_p : float
        Dropout probability for attention weights.
    resid_dropout_p : float
        Dropout probability for residual connections.
    s1_bits : int
        Number of coarse (s1) codebook bits.
    s2_bits : int
        Number of fine (s2) codebook bits.
    beta : float
        Commitment loss weight for BinarySphericalQuantizer.
    gamma0 : float
        BSQ entropy regularisation coefficient (initial).
    gamma : float
        BSQ entropy regularisation coefficient.
    zeta : float
        BSQ temperature parameter.
    group_size : int
        BSQ group size for entropy computation.
    """

    def __init__(
        self,
        d_in: int = 6,
        d_model: int = 128,
        n_heads: int = 8,
        ff_dim: int = 512,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        ffn_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.1,
        resid_dropout_p: float = 0.1,
        s1_bits: int = 10,
        s2_bits: int = 10,
        beta: float = 0.25,
        gamma0: float = 1.0,
        gamma: float = 1.0,
        zeta: float = 0.1,
        group_size: int = 9,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits

        # ── Input projection ──────────────────────────────────────────
        self.embed = nn.Linear(d_in, d_model)

        # ── Encoder stack (n_enc_layers - 1 TransformerBlocks) ────────
        self.encoder = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    ffn_dropout_p=ffn_dropout_p,
                    attn_dropout_p=attn_dropout_p,
                    resid_dropout_p=resid_dropout_p,
                )
                for _ in range(n_enc_layers - 1)
            ]
        )

        # ── Pre-quantization projection ───────────────────────────────
        self.quant_embed = nn.Linear(d_model, self.codebook_dim)

        # ── Binary Spherical Quantizer ────────────────────────────────
        self.tokenizer = BinarySphericalQuantizer(
            embed_dim=self.codebook_dim,
            beta=beta,
            gamma0=gamma0,
            gamma=gamma,
            zeta=zeta,
            group_size=group_size,
        )

        # ── Post-quantization projections ─────────────────────────────
        # s1-only (coarse) path
        self.post_quant_embed_pre = nn.Linear(s1_bits, d_model)
        # full (s1 + s2) path
        self.post_quant_embed = nn.Linear(self.codebook_dim, d_model)

        # ── Decoder stack (n_dec_layers - 1 TransformerBlocks) ────────
        self.decoder = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    ffn_dropout_p=ffn_dropout_p,
                    attn_dropout_p=attn_dropout_p,
                    resid_dropout_p=resid_dropout_p,
                )
                for _ in range(n_dec_layers - 1)
            ]
        )

        # ── Output head ───────────────────────────────────────────────
        self.head = nn.Linear(d_model, d_in)

    # ──────────────────────────────────────────────────────────────────
    # Internal decode helper
    # ──────────────────────────────────────────────────────────────────
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Run decoder stack + output head.

        Parameters
        ----------
        z : Tensor, shape (B, T, d_model)
            Latent representation after post-quantization projection.

        Returns
        -------
        Tensor, shape (B, T, d_in)
            Reconstructed OHLCV features.
        """
        for layer in self.decoder:
            z = layer(z)
        return self.head(z)

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """Full forward pass: encode → quantize → decode (both coarse & full).

        Parameters
        ----------
        x : Tensor, shape (B, T, d_in)
            Input OHLCV features.

        Returns
        -------
        tuple of ((z_pre_recon, z_full_recon), bsq_loss, quantized, z_indices)
            - z_pre_recon : Tensor (B, T, d_in) — reconstruction from s1-only (coarse) tokens
            - z_full_recon : Tensor (B, T, d_in) — reconstruction from full (s1+s2) tokens
            - bsq_loss : Tensor — BinarySphericalQuantizer loss (commitment + entropy)
            - quantized : Tensor (B, T, codebook_dim) — quantized continuous codes
            - z_indices : Tensor (B, T) — integer token indices
        """
        # 1. Input projection
        z = self.embed(x)  # (B, T, d_model)

        # 2. Encoder
        for layer in self.encoder:
            z = layer(z)  # (B, T, d_model)

        # 3. Pre-quantization projection
        z = self.quant_embed(z)  # (B, T, codebook_dim)

        # 4. Quantization
        bsq_loss, quantized, z_indices = self.tokenizer(z)
        # quantized: (B, T, codebook_dim), z_indices: (B, T)

        # 5. Coarse (s1-only) decode path
        quantized_pre = quantized[:, :, : self.s1_bits]  # (B, T, s1_bits)
        z_pre = self.post_quant_embed_pre(quantized_pre)  # (B, T, d_model)
        z_pre_recon = self._decode(z_pre)  # (B, T, d_in)

        # 6. Full (s1 + s2) decode path
        z_full = self.post_quant_embed(quantized)  # (B, T, d_model)
        z_full_recon = self._decode(z_full)  # (B, T, d_in)

        return (z_pre_recon, z_full_recon), bsq_loss, quantized, z_indices

    # ──────────────────────────────────────────────────────────────────
    # Encode
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def encode(self, x: torch.Tensor, half: bool = False) -> torch.Tensor:
        """Encode OHLCV input to discrete token indices.

        Parameters
        ----------
        x : Tensor, shape (B, T, d_in)
            Input OHLCV features.
        half : bool
            If True, return only the coarse (s1) token indices derived
            from the first ``s1_bits`` bits of the full codebook index.

        Returns
        -------
        Tensor, shape (B, T)
            Integer token indices.  When ``half=False`` these index the
            full codebook of size ``2 ** (s1_bits + s2_bits)``; when
            ``half=True`` they index the coarse codebook of size
            ``2 ** s1_bits``.
        """
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)

        # Quantize — pass half to get separate s1/s2 when needed
        _, _, z_indices = self.tokenizer(z, half=half)

        if half:
            # z_indices is a tuple: (s1_indices, s2_indices)
            # each of shape (B, T)
            s1_ids, s2_ids = z_indices
            return s1_ids, s2_ids

        return z_indices

    # ──────────────────────────────────────────────────────────────────
    # Decode
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def decode(self, s1_ids: torch.Tensor, s2_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Decode token indices back to OHLCV space.

        Args:
            s1_ids: (T,) or (B, T) coarse token indices.
            s2_ids: (T,) or (B, T) fine token indices. If None, decode as full indices.
        """
        squeeze_out = s1_ids.ndim == 1
        if squeeze_out:
            s1_ids = s1_ids.unsqueeze(0)
            if s2_ids is not None:
                s2_ids = s2_ids.unsqueeze(0)

        if s2_ids is not None:
            bits_s1 = self.indices_to_bits(s1_ids, half=True)
            bits_s2 = self.indices_to_bits(s2_ids, half=True)
            bits = torch.cat([bits_s1, bits_s2], dim=-1)
            z = self.post_quant_embed(bits)
        else:
            bits = self.indices_to_bits(s1_ids, half=False)
            z = self.post_quant_embed(bits)

        out = self._decode(z)
        if squeeze_out:
            out = out.squeeze(0)
        return out

    # ──────────────────────────────────────────────────────────────────
    # Indices → Bits
    # ──────────────────────────────────────────────────────────────────
    def indices_to_bits(self, x: torch.Tensor, half: bool = False) -> torch.Tensor:
        """Convert integer token indices to bipolar {-1, +1} bit
        representations.

        Each index is decomposed into its binary representation.  Bit
        values {0, 1} are then mapped to bipolar values {-1, +1}.

        Parameters
        ----------
        x : Tensor, shape (B, T)
            Integer token indices.
        half : bool
            If True, interpret ``x`` as coarse (s1-only) indices and
            produce ``s1_bits`` output bits.  If False, interpret ``x``
            as full (s1+s2) indices and produce ``codebook_dim`` output
            bits.

        Returns
        -------
        Tensor, shape (B, T, n_bits)
            Bipolar bit representations where 0 → -1 and 1 → +1.
            ``n_bits`` is ``s1_bits`` when ``half=True``, else
            ``s1_bits + s2_bits``.
        """
        n_bits = self.s1_bits if half else self.codebook_dim

        # Decompose integer index into binary bits (MSB first).
        # x shape: (B, T) → (B, T, n_bits)
        bits = ((x.unsqueeze(-1) >> torch.arange(n_bits - 1, -1, -1, device=x.device)) & 1).float()

        # Map {0, 1} → {-1, +1}
        bits = 2.0 * bits - 1.0

        return bits
