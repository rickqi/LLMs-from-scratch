"""
统一 Predictor API — 4 项目共用接口

用法:
  from projects.shared.predictor import get_predictor
  p = get_predictor("kronos")  # or "medical" / "gbcost"
  result = p.predict(data)
"""

import sys, pickle, torch, numpy as np, pandas as pd, json, logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Project roots
BASE = Path(__file__).resolve().parent.parent
KRONOS_ROOT = BASE / "kronos-stock-predictor"
MEDICAL_ROOT = BASE / "chinese-medical-text-generation"
GBCOST_ROOT = BASE / "gbcost-insurance-ml"


class BasePredictor:
    """统一预测器基类"""
    project: str = "unknown"
    model_info: dict = {}

    def predict(self, input_data) -> dict:
        raise NotImplementedError

    def predict_batch(self, inputs: list) -> list[dict]:
        return [self.predict(d) for d in inputs]


class KronosPredictor(BasePredictor):
    """Kronos 股票预测器 — 方向 + 波动率双输出"""
    project = "kronos"

    def __init__(self):
        sys.path.insert(0, str(KRONOS_ROOT))
        from model.lstm_model import LSTMModel, FEATURES

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.features = FEATURES

        # Direction model
        self.dir_model = LSTMModel(hidden=128, num_layers=2).to(self.device)
        dir_path = KRONOS_ROOT / "outputs/production_model.pt"
        if dir_path.exists():
            self.dir_model.load_state_dict(torch.load(dir_path, map_location=self.device))
        self.dir_model.eval()

        # Volatility model
        self.vol_model = LSTMModel(hidden=128, num_layers=2).to(self.device)
        vol_path = KRONOS_ROOT / "outputs/production_vol_model.pt"
        if vol_path.exists():
            self.vol_model.load_state_dict(torch.load(vol_path, map_location=self.device))
        self.vol_model.eval()

        self.model_info = {
            "direction_rankic": 0.205,
            "volatility_rankic": 0.579,
            "lookback": 180,
            "pred_len": 10,
            "stocks": 68,
        }

    def predict(self, df: pd.DataFrame) -> dict:
        vals = df[self.features].values.astype(np.float32)
        if len(vals) < 180:
            return {"error": "Need 180+ rows"}
        x = vals[-180:]
        mu, std = x.mean(0), x.std(0) + 1e-5
        x_norm = (x - mu) / std
        x_t = torch.FloatTensor(x_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            d = self.dir_model(x_t).item()
            v = self.vol_model(x_t).item()
        return {
            "direction": round(d, 6),
            "direction_label": "UP" if d > 0 else "DOWN",
            "volatility_10d": round(v, 6),
            "last_close": round(float(vals[-1, 3]), 2),
            "composite_signal": round(d * (1 - min(v / 0.1, 1.0)), 6),
        }


class GbcostPredictor(BasePredictor):
    """控费 ML 预测器 — LightGBM 金额预测"""
    project = "gbcost"

    def __init__(self):
        self.model_info = {
            "gini": 0.69,
            "r2": 0.65,
            "model_type": "LightGBM Tweedie",
            "features": 41,
        }

    def predict(self, row: dict) -> dict:
        """单案件预测（需完整特征）"""
        return {
            "predicted_amount": 0.0,
            "status": "LightGBM model loading requires full ml pipeline",
            "note": "Use python -m ml.pipeline.predict for full prediction",
        }


# Registry
PREDICTORS = {
    "kronos": KronosPredictor,
    "gbcost": GbcostPredictor,
}


def get_predictor(name: str) -> BasePredictor:
    if name not in PREDICTORS:
        raise ValueError(f"Unknown project: {name}. Available: {list(PREDICTORS.keys())}")
    return PREDICTORS[name]()


def list_predictors() -> dict:
    result = {}
    for name, cls in PREDICTORS.items():
        try:
            p = cls()
            result[name] = {"project": p.project, "info": p.model_info}
        except Exception as e:
            result[name] = {"error": str(e)}
    return result
