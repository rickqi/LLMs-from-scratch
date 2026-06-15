"""
Kronos 推理脚本

支持三种模式:
  - 单次预测: python inference/generate.py --mode predict --data AAPL.csv
  - 批量预测: python inference/generate.py --mode batch --data_dir ./data/csv/
  - 交互模式: python inference/generate.py --mode interactive
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.default_config import Config
from config.model_configs import build_tokenizer_config, build_model_config
from model.kronos_tokenizer import KronosTokenizer
from model.kronos_model import Kronos
from model.predictor import KronosPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_models(
    tokenizer_path: str,
    predictor_path: str,
    model_size: str = "mini",
    device: str = None,
) -> KronosPredictor:
    """加载 Tokenizer 和 Predictor，返回 KronosPredictor"""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Tokenizer
    tokenizer_cfg = build_tokenizer_config(model_size)
    tokenizer = KronosTokenizer(**tokenizer_cfg)

    if os.path.exists(tokenizer_path):
        ckpt = torch.load(tokenizer_path, map_location="cpu")
        tokenizer.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Tokenizer loaded: {tokenizer_path}")
    else:
        logger.warning(f"Tokenizer not found, using random init: {tokenizer_path}")

    # Predictor
    model_cfg = build_model_config(model_size)
    model = Kronos(**model_cfg)

    if os.path.exists(predictor_path):
        ckpt = torch.load(predictor_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Predictor loaded: {predictor_path}")
    else:
        logger.warning(f"Predictor not found, using random init: {predictor_path}")

    return KronosPredictor(model, tokenizer, device=device)


def generate_forecast(
    predictor: KronosPredictor,
    df: pd.DataFrame,
    pred_len: int = 20,
    T: float = 0.6,
    top_p: float = 0.9,
    sample_count: int = 5,
) -> pd.DataFrame:
    """单次预测"""
    if "timestamps" not in df.columns and df.index.name != "timestamps":
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        else:
            raise ValueError("DataFrame must have 'timestamps' column or DatetimeIndex")
    else:
        timestamps = pd.to_datetime(df["timestamps"] if "timestamps" in df.columns else df.index)
        timestamps = timestamps if "timestamps" not in df.columns else pd.to_datetime(df["timestamps"])

    lookback = min(len(df) - pred_len, predictor.max_context)
    x_df = df.iloc[-lookback:]
    x_ts = timestamps.iloc[-lookback:]

    last_ts = timestamps.iloc[-1]
    freq = pd.infer_freq(timestamps)
    if freq is None:
        freq = "D"
    y_ts = pd.date_range(start=last_ts, periods=pred_len + 1, freq=freq)[1:]

    logger.info(f"Predicting {pred_len} steps ahead (lookback={lookback})")

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=pred_len,
        T=T,
        top_p=top_p,
        sample_count=sample_count,
    )

    return pred_df


def batch_generate(
    predictor: KronosPredictor,
    csv_dir: str,
    pred_len: int = 20,
    T: float = 0.6,
    top_p: float = 0.9,
    sample_count: int = 5,
) -> list[pd.DataFrame]:
    """批量预测目录下所有 CSV 文件"""
    csv_files = sorted(Path(csv_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    logger.info(f"Found {len(csv_files)} CSV files")

    df_list, x_ts_list, y_ts_list = [], [], []
    for f in csv_files:
        df = pd.read_csv(f)
        if "timestamps" in df.columns:
            timestamps = pd.to_datetime(df["timestamps"])
        else:
            timestamps = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(timestamps)

        lookback = min(len(df) - pred_len, predictor.max_context)
        x_df = df.iloc[-lookback:]
        x_ts = timestamps.iloc[-lookback:]

        freq = pd.infer_freq(timestamps) or "D"
        y_ts = pd.date_range(start=timestamps.iloc[-1], periods=pred_len + 1, freq=freq)[1:]

        df_list.append(x_df)
        x_ts_list.append(x_ts)
        y_ts_list.append(y_ts)

    results = predictor.predict_batch(
        df_list=df_list,
        x_timestamp_list=x_ts_list,
        y_timestamp_list=y_ts_list,
        pred_len=pred_len,
        T=T,
        top_p=top_p,
        sample_count=sample_count,
    )

    return results


def interactive_mode(predictor: KronosPredictor):
    """交互式预测"""
    print("\n" + "=" * 60)
    print("  Kronos Stock Predictor - Interactive Mode")
    print("  Type 'quit' to exit, 'help' for commands")
    print("=" * 60 + "\n")

    while True:
        try:
            cmd = input("kronos> ").strip()
            if cmd.lower() in ("quit", "exit", "q"):
                break
            elif cmd.lower() == "help":
                print("Commands:")
                print("  predict <symbol> <lookback> <pred_len>  - Make prediction")
                print("  config                               - Show current config")
                print("  quit                                 - Exit")
            elif cmd.lower() == "config":
                print(f"  max_context: {predictor.max_context}")
                print(f"  device: {predictor.device}")
            elif cmd.startswith("predict"):
                parts = cmd.split()
                if len(parts) >= 3:
                    symbol = parts[1]
                    pred_len = int(parts[2]) if len(parts) > 2 else 20
                    print(f"  Predicting {symbol} for {pred_len} steps...")
                    print("  (Requires data file - use generate_forecast API)")
                else:
                    print("  Usage: predict <symbol> <pred_len>")
            else:
                print(f"  Unknown command: {cmd}")
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            break
        except Exception as e:
            print(f"  Error: {e}")

    print("Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="Kronos Stock Predictor Inference")
    parser.add_argument("--mode", type=str, default="predict",
                        choices=["predict", "batch", "interactive"])
    parser.add_argument("--tokenizer_path", type=str,
                        default="./outputs/tokenizer/best_model.pt")
    parser.add_argument("--predictor_path", type=str,
                        default="./outputs/predictor/best_model.pt")
    parser.add_argument("--model_size", type=str, default="mini")
    parser.add_argument("--data", type=str, default=None, help="CSV file or directory")
    parser.add_argument("--pred_len", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--sample_count", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    predictor = load_models(
        args.tokenizer_path, args.predictor_path,
        model_size=args.model_size, device=args.device,
    )

    if args.mode == "predict":
        if not args.data:
            parser.error("--data required for predict mode")
        df = pd.read_csv(args.data)
        pred_df = generate_forecast(
            predictor, df,
            pred_len=args.pred_len,
            T=args.temperature,
            top_p=args.top_p,
            sample_count=args.sample_count,
        )
        print(pred_df)
        if args.output:
            pred_df.to_csv(args.output)
            logger.info(f"Predictions saved to: {args.output}")

    elif args.mode == "batch":
        if not args.data:
            parser.error("--data (directory) required for batch mode")
        results = batch_generate(
            predictor, args.data,
            pred_len=args.pred_len,
            T=args.temperature,
            top_p=args.top_p,
            sample_count=args.sample_count,
        )
        for i, pred_df in enumerate(results):
            print(f"\n--- Prediction {i+1} ---")
            print(pred_df.head())

    elif args.mode == "interactive":
        interactive_mode(predictor)


if __name__ == "__main__":
    main()
