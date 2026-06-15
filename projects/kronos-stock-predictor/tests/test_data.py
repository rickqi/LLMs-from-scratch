"""
数据管道测试
"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestSymbols:
    def test_symbols_list(self):
        from data.symbols import CSI300_SYMBOLS, get_default_symbols
        assert len(CSI300_SYMBOLS) > 0
        symbols = get_default_symbols()
        assert len(symbols) > 0

    def test_tushare_to_qlib(self):
        from data.symbols import tushare_to_qlib, qlib_to_tushare
        assert tushare_to_qlib("600000.SH") == "sh600000"
        assert tushare_to_qlib("000001.SZ") == "sz000001"
        assert qlib_to_tushare("sh600000") == "600000.SH"


class TestConfig:
    def test_default_config(self):
        from config.default_config import Config
        config = Config()
        assert config.lookback_window == 90
        assert config.predict_window == 10

    def test_model_configs(self):
        from config.model_configs import get_model_config, build_tokenizer_config, build_model_config
        mini = get_model_config("mini")
        assert mini["params"] == "4.1M"

        tok_cfg = build_tokenizer_config("mini")
        assert tok_cfg["d_model"] == 64
        assert tok_cfg["s1_bits"] == 10

        model_cfg = build_model_config("mini")
        assert model_cfg["n_layers"] == 4


class TestLosses:
    def test_tokenizer_loss(self):
        import torch
        from train.losses import tokenizer_loss
        x = torch.randn(2, 10, 6)
        z_pre = torch.randn(2, 10, 6)
        z_full = torch.randn(2, 10, 6)
        bsq_loss = torch.tensor(0.1)
        loss = tokenizer_loss(x, z_pre, z_full, bsq_loss)
        assert loss.item() > 0

    def test_predictor_loss(self):
        import torch
        from train.losses import predictor_loss
        s1_logits = torch.randn(2, 10, 1024)
        s2_logits = torch.randn(2, 10, 1024)
        s1_targets = torch.randint(0, 1024, (2, 10))
        s2_targets = torch.randint(0, 1024, (2, 10))
        total, l1, l2 = predictor_loss(s1_logits, s2_logits, s1_targets, s2_targets)
        assert total.item() > 0
