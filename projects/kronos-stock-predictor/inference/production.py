"""
Kronos 生产预测模块 — 加载最优 LSTM 模型，提供统一预测 API

最优配置:
  模型: LSTM (128h, 2层)
  数据: 68 半导体 + 长历史 (2005-2026)
  参数: lookback=180, pred_len=10
  RankIC: +0.205
"""

import pickle, torch, numpy as np, pandas as pd, logging
from pathlib import Path
from model.lstm_model import LSTMModel, FEATURES

logger = logging.getLogger(__name__)

OPTIMAL_CONFIG = {
    "model": "LSTM",
    "hidden": 128,
    "num_layers": 2,
    "lookback": 180,
    "pred_len": 10,
    "features": FEATURES,  # ['open','high','low','close','vol','amt']
    "rankic": 0.205,
    "dir_acc": 0.593,
    "data": "68 semiconductor stocks + 7 long-history (2005-2026)",
}


class Predictor:
    """生产级预测器"""
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lookback = OPTIMAL_CONFIG["lookback"]
        self.pred_len = OPTIMAL_CONFIG["pred_len"]
        self.features = OPTIMAL_CONFIG["features"]

        self.model = LSTMModel(
            hidden=OPTIMAL_CONFIG["hidden"],
            num_layers=OPTIMAL_CONFIG["num_layers"],
        ).to(self.device)

        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, df: pd.DataFrame) -> dict:
        """预测单只股票未来 pred_len 日收益"""
        vals = df[self.features].values.astype(np.float32)
        if len(vals) < self.lookback:
            return {"error": f"Need {self.lookback} rows, got {len(vals)}"}

        x = vals[-self.lookback:]
        mean, std = x.mean(axis=0), x.std(axis=0) + 1e-5
        x_norm = (x - mean) / std

        with torch.no_grad():
            pred = self.model(
                torch.FloatTensor(x_norm).unsqueeze(0).to(self.device)
            ).item()

        last_close = vals[-1, 3]
        return {
            "predicted_return": float(pred),
            "direction": "UP" if pred > 0 else "DOWN",
            "confidence": abs(pred),
            "last_close": float(last_close),
            "target_close": float(last_close * (1 + pred)),
            "pred_len_days": self.pred_len,
        }

    def predict_batch(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """批量预测多只股票，返回排序后的 DataFrame"""
        results = []
        for sym, df in data.items():
            r = self.predict(df)
            if "error" not in r:
                r["symbol"] = sym
                results.append(r)
        return pd.DataFrame(results).sort_values("confidence", ascending=False)
