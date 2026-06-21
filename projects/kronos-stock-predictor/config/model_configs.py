"""
模型规模配置 — 参考 Kronos 论文 Model Zoo
"""

# 所有模型共享的分词器参数
_TOKENIZER_COMMON = dict(
    d_in=6,              # OHLCVA, set to 14 for factor-enhanced mode
    n_heads=8,
    ff_dim=512,
    n_enc_layers=4,
    n_dec_layers=4,
    ffn_dropout_p=0.1,
    attn_dropout_p=0.1,
    resid_dropout_p=0.1,
    s1_bits=10,          # 粗粒度 token: 10 bits → vocab_size=1024
    s2_bits=10,          # 细粒度 token: 10 bits → vocab_size=1024
    beta=0.25,           # BSQ commitment loss 权重
    gamma0=1.0,
    gamma=1.0,
    zeta=0.1,
    group_size=10,
)

# 模型规模配置
MODEL_CONFIGS = {
    "mini": {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 4,
        "ff_dim": 256,
        "s1_bits": 10,
        "s2_bits": 10,
        "ffn_dropout_p": 0.1,
        "attn_dropout_p": 0.1,
        "resid_dropout_p": 0.1,
        "token_dropout_p": 0.1,
        "learn_te": True,
        "context_len": 2048,
        "params": "4.1M",
        # MoE settings (set use_moe=True to enable)
        "use_moe": False,
        "num_experts": 4,
        "moe_top_k": 2,
        "tokenizer_d_model": 64,      # Tokenizer 内部的 d_model
        "tokenizer_ff_dim": 256,
        "tokenizer_n_heads": 4,
        "tokenizer_n_enc_layers": 3,
        "tokenizer_n_dec_layers": 3,
    },
    "small": {
        "d_model": 192,
        "n_heads": 6,
        "n_layers": 8,
        "ff_dim": 768,
        "s1_bits": 10,
        "s2_bits": 10,
        "ffn_dropout_p": 0.1,
        "attn_dropout_p": 0.1,
        "resid_dropout_p": 0.1,
        "token_dropout_p": 0.1,
        "learn_te": True,
        "context_len": 512,
        "params": "28M",
        "tokenizer_d_model": 128,
        "tokenizer_ff_dim": 512,
        "tokenizer_n_heads": 8,
        "tokenizer_n_enc_layers": 4,
        "tokenizer_n_dec_layers": 4,
    },
    "base": {
        "d_model": 384,
        "n_heads": 12,
        "n_layers": 12,
        "ff_dim": 1536,
        "s1_bits": 10,
        "s2_bits": 10,
        "ffn_dropout_p": 0.1,
        "attn_dropout_p": 0.1,
        "resid_dropout_p": 0.1,
        "token_dropout_p": 0.1,
        "learn_te": True,
        "context_len": 512,
        "params": "103M",
        "tokenizer_d_model": 256,
        "tokenizer_ff_dim": 1024,
        "tokenizer_n_heads": 8,
        "tokenizer_n_enc_layers": 4,
        "tokenizer_n_dec_layers": 4,
    },
}


def get_model_config(name: str) -> dict:
    """获取指定规模的模型配置"""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model config '{name}'. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[name]


def build_tokenizer_config(model_size: str) -> dict:
    """基于模型规模构建 Tokenizer 配置"""
    mc = get_model_config(model_size)
    return {
        **_TOKENIZER_COMMON,
        "d_model": mc["tokenizer_d_model"],
        "n_heads": mc["tokenizer_n_heads"],
        "ff_dim": mc["tokenizer_ff_dim"],
        "n_enc_layers": mc["tokenizer_n_enc_layers"],
        "n_dec_layers": mc["tokenizer_n_dec_layers"],
    }


def build_model_config(model_size: str) -> dict:
    """基于模型规模构建 Predictor 配置"""
    mc = get_model_config(model_size)
    return {
        "s1_bits": mc["s1_bits"],
        "s2_bits": mc["s2_bits"],
        "n_layers": mc["n_layers"],
        "d_model": mc["d_model"],
        "n_heads": mc["n_heads"],
        "ff_dim": mc["ff_dim"],
        "ffn_dropout_p": mc["ffn_dropout_p"],
        "attn_dropout_p": mc["attn_dropout_p"],
        "resid_dropout_p": mc["resid_dropout_p"],
        "token_dropout_p": mc["token_dropout_p"],
        "learn_te": mc["learn_te"],
    }
