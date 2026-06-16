"""
全局配置类 — 参考 Kronos/finetune/config.py + TradingAgents/default_config.py
"""

import os


class Config:
    """Kronos 股票预测项目全局配置"""

    def __init__(self):
        # =================================================================
        # 数据参数
        # =================================================================
        self.instrument = "csi300"
        self.feature_list = ["open", "high", "low", "close", "vol", "amt"]
        self.time_feature_list = ["minute", "hour", "weekday", "day", "month"]

        # 数据时间范围
        self.dataset_begin_time = "2015-01-01"
        self.dataset_end_time = "2025-06-30"

        # 滑动窗口参数
        self.lookback_window = 90
        self.predict_window = 10
        self.max_context = 512
        self.clip = 5.0

        # =================================================================
        # 数据集划分
        # =================================================================
        self.train_time_range = ["2015-01-01", "2022-12-31"]
        self.val_time_range = ["2023-01-01", "2023-12-31"]
        self.test_time_range = ["2024-01-01", "2025-06-30"]
        self.backtest_time_range = ["2024-07-01", "2025-06-30"]

        # 预处理数据路径
        self.dataset_path = "./data/processed"

        # =================================================================
        # 训练超参数
        # =================================================================
        self.epochs = 30
        self.log_interval = 100
        self.batch_size = 50

        self.n_train_iter = 2000 * self.batch_size
        self.n_val_iter = 400 * self.batch_size

        # 学习率
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 4e-5

        # 梯度累积
        self.accumulation_steps = 1

        # AdamW 优化器参数
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1

        # 随机种子
        self.seed = 100

        # =================================================================
        # 模型路径
        # =================================================================
        self.save_path = "./outputs"
        self.tokenizer_save_folder = "tokenizer"
        self.predictor_save_folder = "predictor"

        self.pretrained_tokenizer_path = None
        self.pretrained_predictor_path = None

        # =================================================================
        # 推理参数
        # =================================================================
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5

        # =================================================================
        # 回测参数
        # =================================================================
        self.backtest_n_symbol_hold = 50
        self.backtest_n_symbol_drop = 5
        self.backtest_hold_thresh = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = "SH000300"  # 沪深300

        # =================================================================
        # 日志
        # =================================================================
        self.use_comet = False
        self.comet_config = {}

    def to_dict(self) -> dict:
        """将配置转为字典（用于分布式训练传递）"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        config = cls()
        for k, v in d.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config
