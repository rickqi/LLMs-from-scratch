"""
Kronos 模型测试
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestModules:
    """测试基础模块"""

    def test_rms_norm(self):
        from model.modules import RMSNorm
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        y = norm(x)
        assert y.shape == x.shape

    def test_transformer_block(self):
        from model.modules import TransformerBlock
        block = TransformerBlock(d_model=64, n_heads=4, ff_dim=256,
                                 ffn_dropout_p=0.1, attn_dropout_p=0.1, resid_dropout_p=0.1)
        x = torch.randn(2, 10, 64)
        y = block(x)
        assert y.shape == x.shape

    def test_bsq_quantizer(self):
        from model.modules import BinarySphericalQuantizer
        bsq = BinarySphericalQuantizer(embed_dim=20, beta=0.25, gamma0=1.0, gamma=1.0, zeta=0.1, group_size=10)
        z = torch.randn(2, 10, 20)
        loss, quantized, indices = bsq(z, collect_metrics=False)
        assert quantized.shape == (2, 10, 20)

    def test_hierarchical_embedding(self):
        from model.modules import HierarchicalEmbedding
        emb = HierarchicalEmbedding(s1_bits=10, s2_bits=10, d_model=64)
        s1_ids = torch.randint(0, 2 ** 10, (2, 10))
        s2_ids = torch.randint(0, 2 ** 10, (2, 10))
        y = emb([s1_ids, s2_ids])
        assert y.shape == (2, 10, 64)


class TestTokenizer:
    """测试 Tokenizer"""

    def test_tokenizer_forward(self):
        from model.modules import TransformerBlock, BinarySphericalQuantizer
        from model.kronos_tokenizer import KronosTokenizer

        tokenizer = KronosTokenizer(
            d_in=6, d_model=64, n_heads=4, ff_dim=256,
            n_enc_layers=3, n_dec_layers=3,
            ffn_dropout_p=0.1, attn_dropout_p=0.1, resid_dropout_p=0.1,
            s1_bits=10, s2_bits=10,
            beta=0.25, gamma0=1.0, gamma=1.0, zeta=0.1, group_size=10,
        )
        x = torch.randn(2, 20, 6)
        (z_pre, z_full), bsq_loss, quantized, indices = tokenizer(x)
        assert z_pre.shape == (2, 20, 6)
        assert z_full.shape == (2, 20, 6)

    def test_tokenizer_encode_decode(self):
        from model.kronos_tokenizer import KronosTokenizer
        tokenizer = KronosTokenizer(
            d_in=6, d_model=64, n_heads=4, ff_dim=256,
            n_enc_layers=3, n_dec_layers=3,
            ffn_dropout_p=0.1, attn_dropout_p=0.1, resid_dropout_p=0.1,
            s1_bits=10, s2_bits=10,
            beta=0.25, gamma0=1.0, gamma=1.0, zeta=0.1, group_size=10,
        )
        x = torch.randn(2, 20, 6)
        indices = tokenizer.encode(x, half=True)
        reconstructed = tokenizer.decode(indices, half=True)
        assert reconstructed.shape == (2, 20, 6)


class TestModel:
    """测试 Kronos 模型"""

    def test_model_forward(self):
        from model.kronos_model import Kronos
        model = Kronos(
            s1_bits=10, s2_bits=10,
            n_layers=2, d_model=64, n_heads=4, ff_dim=256,
            ffn_dropout_p=0.1, attn_dropout_p=0.1, resid_dropout_p=0.1,
            token_dropout_p=0.1, learn_te=True,
        )
        s1_ids = torch.randint(0, 1024, (2, 20))
        s2_ids = torch.randint(0, 1024, (2, 20))
        stamp = torch.randint(0, 60, (2, 20, 5)).float()

        s1_logits, s2_logits = model(s1_ids, s2_ids, stamp=stamp)
        assert s1_logits.shape == (2, 20, 1024)
        assert s2_logits.shape == (2, 20, 1024)

    def test_model_decode_s1_s2(self):
        from model.kronos_model import Kronos
        model = Kronos(
            s1_bits=10, s2_bits=10,
            n_layers=2, d_model=64, n_heads=4, ff_dim=256,
            ffn_dropout_p=0.1, attn_dropout_p=0.1, resid_dropout_p=0.1,
            token_dropout_p=0.1, learn_te=True,
        )
        s1_ids = torch.randint(0, 1024, (2, 20))
        s2_ids = torch.randint(0, 1024, (2, 20))
        stamp = torch.randint(0, 60, (2, 20, 5)).float()

        s1_logits, context = model.decode_s1(s1_ids, s2_ids, stamp=stamp)
        s2_logits = model.decode_s2(context, s1_ids)
        assert s1_logits.shape == (2, 20, 1024)
        assert s2_logits.shape == (2, 20, 1024)


class TestPredictor:
    """测试 Predictor"""

    def test_predictor_init(self):
        from model.kronos_tokenizer import KronosTokenizer
        from model.kronos_model import Kronos
        from model.predictor import KronosPredictor

        tokenizer = KronosTokenizer(
            d_in=6, d_model=64, n_heads=4, ff_dim=256,
            n_enc_layers=2, n_dec_layers=2,
            ffn_dropout_p=0.1, attn_dropout_p=0.1, resid_dropout_p=0.1,
            s1_bits=10, s2_bits=10,
            beta=0.25, gamma0=1.0, gamma=1.0, zeta=0.1, group_size=10,
        )
        model = Kronos(
            s1_bits=10, s2_bits=10,
            n_layers=2, d_model=64, n_heads=4, ff_dim=256,
            ffn_dropout_p=0.1, attn_dropout_p=0.1, resid_dropout_p=0.1,
            token_dropout_p=0.1, learn_te=True,
        )
        predictor = KronosPredictor(model, tokenizer, device="cpu")
        assert str(predictor.device) == "cpu"
        assert predictor.max_context == 512
