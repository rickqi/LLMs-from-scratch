"""
Kronos — Decoder-only Transformer for autoregressive prediction on hierarchical discrete tokens.

Architecture overview:
    1. HierarchicalEmbedding maps (s1_ids, s2_ids) → d_model vectors
    2. Optional TemporalEmbedding adds time-stamp information
    3. Stack of TransformerBlock layers with causal attention
    4. RMSNorm → DualHead produces s1 logits
    5. DependencyAwareLayer conditions on sampled s1 → s2 logits via DualHead.cond_forward

During inference, `auto_regressive_inference` orchestrates the sliding-window generation
loop: predict s1 (coarse), sample, predict s2 (fine) conditioned on s1, sample, append.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from tqdm import tqdm

from model.modules import (
    TransformerBlock,
    RMSNorm,
    HierarchicalEmbedding,
    TemporalEmbedding,
    DependencyAwareLayer,
    DualHead,
)


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """Filter a logits distribution using top-k and/or nucleus (top-p) filtering.

    Args:
        logits: logits distribution shape (batch_size, vocab_size)
        top_k:  keep only top k tokens with highest probability (0 = disabled)
        top_p:  keep the smallest set of tokens whose cumulative probability >= top_p (0 = disabled)

    Returns:
        Filtered logits with the same shape, where invalid positions are set to -inf.
    """
    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # safety clamp
        # Remove tokens with probability less than the k-th highest
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        # Shift right so that at least one token is always kept
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    sample_logits: bool = True,
) -> torch.Tensor:
    """Sample token indices from a logits tensor.

    Args:
        logits:        (..., vocab_size) unnormalised log-probabilities
        temperature:   scaling factor; higher → more random
        top_k:         top-k filtering parameter (0 = disabled)
        top_p:         nucleus filtering parameter (0 = disabled)
        sample_logits: if False, take argmax instead of sampling

    Returns:
        (batch,) LongTensor of sampled token indices
    """
    if temperature != 1.0:
        logits = logits / temperature

    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    if not sample_logits:
        return logits.argmax(dim=-1)

    probs = F.softmax(logits, dim=-1)
    # Handle both 2D (B, V) and 3D (B, T, V) inputs
    orig_shape = probs.shape
    probs_2d = probs.reshape(-1, orig_shape[-1])
    samples = torch.multinomial(probs_2d, num_samples=1)
    return samples.reshape(orig_shape[:-1])


# ---------------------------------------------------------------------------
# Time-stamp helper
# ---------------------------------------------------------------------------

def calc_time_stamps(x_timestamp: pd.DatetimeIndex) -> pd.DataFrame:
    """Convert a DatetimeIndex into a DataFrame of calendar features.

    Args:
        x_timestamp: pandas DatetimeIndex

    Returns:
        DataFrame with columns ['minute', 'hour', 'weekday', 'day', 'month']
    """
    return pd.DataFrame({
        "minute":  x_timestamp.minute.values,
        "hour":    x_timestamp.hour.values,
        "weekday": x_timestamp.weekday.values,
        "day":     x_timestamp.day.values,
        "month":   x_timestamp.month.values,
    }, index=x_timestamp)

    # Accept Series by converting to DatetimeIndex first
    if isinstance(x_timestamp, (pd.Series, pd.DatetimeIndex)):
        x_timestamp = pd.DatetimeIndex(x_timestamp)
    elif hasattr(x_timestamp, 'to_datetime'):
        x_timestamp = pd.DatetimeIndex(pd.to_datetime(x_timestamp))

    return pd.DataFrame({
        "minute":  x_timestamp.minute,
        "hour":    x_timestamp.hour,
        "weekday": x_timestamp.weekday,
        "day":     x_timestamp.day,
        "month":   x_timestamp.month,
    }, index=x_timestamp)


# ---------------------------------------------------------------------------
# Kronos model
# ---------------------------------------------------------------------------

class Kronos(nn.Module, PyTorchModelHubMixin):
    """Decoder-only Transformer with hierarchical (s1, s2) token prediction.

    The model first predicts coarse s1 tokens, then conditions on the sampled
    s1 to predict fine s2 tokens — a two-level autoregressive decomposition.
    """

    def __init__(
        self,
        s1_bits: int = 10,
        s2_bits: int = 10,
        n_layers: int = 12,
        d_model: int = 384,
        n_heads: int = 12,
        ff_dim: int = 1536,
        ffn_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.1,
        resid_dropout_p: float = 0.1,
        token_dropout_p: float = 0.1,
        learn_te: bool = True,
    ):
        super().__init__()

        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.token_dropout_p = token_dropout_p
        self.learn_te = learn_te

        self.s1_vocab_size = 2 ** s1_bits
        self.s2_vocab_size = 2 ** s2_bits

        # Embeddings
        self.embedding = HierarchicalEmbedding(s1_bits, s2_bits, d_model)
        self.time_emb = TemporalEmbedding(d_model, learn_te)
        self.token_drop = nn.Dropout(token_dropout_p)

        # Transformer backbone
        self.transformer = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

        # Hierarchical prediction heads
        self.dep_layer = DependencyAwareLayer(d_model)
        self.head = DualHead(s1_bits, s2_bits, d_model)

        # Weight initialisation
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        s1_ids: torch.LongTensor,
        s2_ids: torch.LongTensor,
        stamp: torch.Tensor | None = None,
        padding_mask: torch.BoolTensor | None = None,
        use_teacher_forcing: bool = False,
        s1_targets: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the full hierarchical forward pass.

        Args:
            s1_ids:              (B, T) coarse token ids
            s2_ids:              (B, T) fine token ids
            stamp:               (B, T, F) temporal features, or None
            padding_mask:        (B, T) True for positions to ignore
            use_teacher_forcing: if True, use ground-truth s1_targets for s2 conditioning
            s1_targets:          (B, T) ground-truth s1 ids (required when use_teacher_forcing)

        Returns:
            s1_logits: (B, T, s1_vocab_size)
            s2_logits: (B, T, s2_vocab_size)
        """
        # 1. Embed tokens
        x = self.embedding([s1_ids, s2_ids])

        # 2. Add temporal embedding
        if stamp is not None:
            x = x + self.time_emb(stamp)

        # 3. Token dropout
        x = self.token_drop(x)

        # 4. Transformer layers
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        # 5. Final norm
        x = self.norm(x)

        # 6. s1 logits
        s1_logits = self.head(x)

        # 7. Obtain sibling embedding for s2 conditioning
        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_sampled = sample_from_logits(s1_logits.detach())
            sibling_embed = self.embedding.emb_s1(s1_sampled)

        # 8. Dependency-aware conditioning
        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)

        # 9. s2 logits
        s2_logits = self.head.cond_forward(x2)

        return s1_logits, s2_logits

    # ------------------------------------------------------------------
    # Decoupled decode helpers (used during inference)
    # ------------------------------------------------------------------

    def decode_s1(
        self,
        s1_ids: torch.LongTensor,
        s2_ids: torch.LongTensor,
        stamp: torch.Tensor | None = None,
        padding_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the transformer backbone and produce s1 logits + context.

        This is the first half of the two-stage decode.  The returned
        *context* can be fed into ``decode_s2`` to obtain s2 predictions
        without re-running the transformer.

        Args:
            s1_ids:       (B, T) coarse token ids
            s2_ids:       (B, T) fine token ids
            stamp:        (B, T, F) temporal features, or None
            padding_mask: (B, T) True for positions to ignore

        Returns:
            s1_logits: (B, T, s1_vocab_size)
            context:   (B, T, d_model)  — transformer output before the head
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        x = self.token_drop(x)
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)

        s1_logits = self.head(x)
        return s1_logits, x

    def decode_s2(
        self,
        context: torch.Tensor,
        s1_ids: torch.LongTensor,
        padding_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """Produce s2 logits conditioned on a pre-computed context and sampled s1.

        Args:
            context:      (B, T, d_model) from ``decode_s1``
            s1_ids:       (B, T) sampled coarse token ids
            padding_mask: (B, T) True for positions to ignore

        Returns:
            s2_logits: (B, T, s2_vocab_size)
        """
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        s2_logits = self.head.cond_forward(x2)
        return s2_logits

    # ------------------------------------------------------------------
    # Auto-regressive inference
    # ------------------------------------------------------------------


def auto_regressive_inference(
    tokenizer,
    model: Kronos,
    x: dict[str, torch.LongTensor],
    x_stamp: torch.Tensor | None,
    y_stamp: torch.Tensor | None,
    max_context: int,
    pred_len: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    sample_logits: bool = True,
    sample_count: int = 1,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> np.ndarray:
    """Run auto-regressive inference with a sliding context window.

    For each prediction step:
        1. decode_s1 → sample s1
        2. decode_s2 → sample s2
        3. Append (s1, s2) to the buffer and slide the window

    When ``sample_count > 1``, the input is duplicated *sample_count* times
    and multiple independent trajectories are generated.  The final result
    is the average across trajectories.

    Args:
        tokenizer:    KronosTokenizer with ``decode()`` method
        model:        Kronos model instance
        x:            dict with keys 's1_ids' and 's2_ids', each (1, T_ctx)
        x_stamp:      (1, T_ctx, F) temporal features for context, or None
        y_stamp:      (1, T_pred, F) temporal features for prediction window, or None
        max_context:  maximum context length (sliding window size)
        pred_len:     number of future steps to predict
        temperature:  sampling temperature
        top_k:        top-k filtering (0 = disabled)
        top_p:        nucleus filtering (0 = disabled)
        sample_logits: if False, use greedy argmax
        sample_count: number of independent trajectories to average
        device:       torch device
        verbose:      show tqdm progress bar

    Returns:
        numpy array of shape (pred_len, d_in) — predicted OHLCV values
    """
    model.eval()

    # ---- Prepare input buffers (duplicate for multiple trajectories) ----
    s1_buf = x["s1_ids"].repeat(sample_count, 1).to(device)   # (S, T_ctx)
    s2_buf = x["s2_ids"].repeat(sample_count, 1).to(device)   # (S, T_ctx)

    if x_stamp is not None:
        x_stamp_rep = x_stamp.repeat(sample_count, 1, 1).to(device)
    else:
        x_stamp_rep = None

    # Pre-compute y_stamp repeated
    if y_stamp is not None:
        y_stamp_rep = y_stamp.repeat(sample_count, 1, 1).to(device)  # (S, pred_len, F)
    else:
        y_stamp_rep = None

    # ---- Pre-compute full stamp ----
    if x_stamp_rep is not None and y_stamp_rep is not None:
        full_stamp = torch.cat([x_stamp_rep, y_stamp_rep], dim=1)  # (S, T_ctx+pred_len, F)
    elif x_stamp_rep is not None:
        full_stamp = x_stamp_rep
    else:
        full_stamp = None

    # ---- Auto-regressive loop ----
    pred_s1_list: list[torch.LongTensor] = []
    pred_s2_list: list[torch.LongTensor] = []
    T_ctx = s1_buf.size(1)

    with torch.no_grad():
        for t in tqdm(range(pred_len), disable=not verbose, desc="Autoregressive"):
            # Build stamp matching current buffer length
            total_len = T_ctx + t
            if full_stamp is not None:
                start = max(0, total_len - max_context)
                end = total_len
                cur_stamp = full_stamp[:, start:end, :]
            else:
                cur_stamp = None

            # Step 1: decode_s1
            s1_logits, context = model.decode_s1(s1_buf, s2_buf, stamp=cur_stamp)
            # Take logits at the last position
            last_s1_logits = s1_logits[:, -1, :]  # (S, s1_vocab_size)
            s1_sampled = sample_from_logits(
                last_s1_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample_logits=sample_logits,
            )  # (S,)

            # Step 2: decode_s2 using the sampled s1
            # Build s1_ids for decode_s2: use the full buffer but replace last position
            # Actually we just need the context + the sampled s1 at the last position
            # The dep_layer operates on the full context, so we pass the full s1_buf
            # with the last position replaced by the sampled s1
            s1_for_s2 = s1_buf.clone()
            s1_for_s2[:, -1] = s1_sampled

            s2_logits = model.decode_s2(context, s1_for_s2)
            last_s2_logits = s2_logits[:, -1, :]  # (S, s2_vocab_size)
            s2_sampled = sample_from_logits(
                last_s2_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample_logits=sample_logits,
            )  # (S,)

            # Record predictions
            pred_s1_list.append(s1_sampled.unsqueeze(1))  # (S, 1)
            pred_s2_list.append(s2_sampled.unsqueeze(1))  # (S, 1)

            # Append to buffer and enforce sliding window
            s1_buf = torch.cat([s1_buf, s1_sampled.unsqueeze(1)], dim=1)  # (S, T+1)
            s2_buf = torch.cat([s2_buf, s2_sampled.unsqueeze(1)], dim=1)  # (S, T+1)

            # Slide window if exceeding max_context
            if s1_buf.size(1) > max_context:
                s1_buf = s1_buf[:, -max_context:]
                s2_buf = s2_buf[:, -max_context:]
                # Trim x_stamp_rep to match the new buffer length
                if x_stamp_rep is not None:
                    # Buffer now has max_context tokens; cur_stamp must match
                    # After sliding, the effective context is the last max_context tokens
                    # which correspond to stamps from (total_len - max_context) onward
                    pass  # Stamp rebuilt per iteration, handled in cur_stamp construction

    # ---- Assemble predicted token sequences ----
    pred_s1 = torch.cat(pred_s1_list, dim=1)  # (S, pred_len)
    pred_s2 = torch.cat(pred_s2_list, dim=1)  # (S, pred_len)

    # ---- Decode tokens back to OHLCV via tokenizer ----
    # Average across sample trajectories
    results = []
    for i in range(sample_count):
        decoded = tokenizer.decode(pred_s1[i], pred_s2[i]).cpu()
        results.append(decoded)

    # Stack and average: (sample_count, pred_len, d_in) → (pred_len, d_in)
    averaged = np.mean(np.stack(results, axis=0), axis=0)
    return averaged
