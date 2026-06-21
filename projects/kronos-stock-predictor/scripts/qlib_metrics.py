#!/usr/bin/env python3
"""
Kronos Qlib-Compatible Backtest Engine
======================================
无需 Microsoft Qlib 依赖，输出相同的标准化指标:
  AR (Annual Return)   IR (Information Ratio)
  IC (Information Coefficient)  ICIR (IC / IC_std)
  MaxDD (Maximum Drawdown)  Sharpe Ratio
  WinRate  Turnover

用法:
    python scripts/qlib_metrics.py \
        --predictions ./outputs/predictions.npy \
        --benchmark SH000300 \
        --output ./outputs/qlib_results.json
"""

import numpy as np
import pandas as pd
import json, argparse, logging
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


class QlibMetrics:
    """Qlib 标准评估指标计算"""
    
    def __init__(self, predictions: np.ndarray, returns: np.ndarray, 
                 dates: np.ndarray | None = None, benchmark_returns: np.ndarray | None = None):
        """
        Args:
            predictions: (N, T) or (N,) 预测值 (截面 or 时序)
            returns:     (N, T) or (N,) 实际收益
            dates:       (T,) 日期索引
            benchmark_returns: (T,) 基准收益
        """
        self.pred = np.asarray(predictions)
        self.ret = np.asarray(returns)
        self.dates = np.asarray(dates) if dates is not None else None
        self.bench_ret = np.asarray(benchmark_returns) if benchmark_returns is not None else None
    
    def compute_all(self) -> dict:
        """计算全部 Qlib 标准指标"""
        metrics = {}
        metrics.update(self._ic_metrics())
        metrics.update(self._portfolio_metrics())
        metrics.update(self._risk_metrics())
        return metrics
    
    def _ic_metrics(self) -> dict:
        """IC (Information Coefficient) 系列指标"""
        pred = self.pred
        ret = self.ret
        
        if pred.ndim == 2 and ret.ndim == 2:
            # 截面: (N_stocks, T_days)
            ics = []
            for t in range(pred.shape[1]):
                mask = ~(np.isnan(pred[:, t]) | np.isnan(ret[:, t]))
                if mask.sum() >= 10:
                    ic, _ = spearmanr(pred[:, t][mask], ret[:, t][mask])
                    ics.append(ic if not np.isnan(ic) else 0)
            ics = np.array(ics)
        else:
            # 时序: 单资产
            mask = ~(np.isnan(pred) | np.isnan(ret))
            ic, _ = spearmanr(pred[mask], ret[mask])
            ics = np.array([ic])
        
        if len(ics) == 0:
            return {'IC_mean': 0, 'IC_std': 0, 'ICIR': 0, 'RankIC': 0}
        
        ic_mean = np.mean(ics)
        ic_std = np.std(ics) + 1e-8
        icir = ic_mean / ic_std
        
        return {
            'IC_mean': float(ic_mean),
            'IC_std': float(ic_std),
            'ICIR': float(icir),
            'RankIC': float(ic_mean),  # 使用 Spearman = RankIC
        }
    
    def _portfolio_metrics(self) -> dict:
        """组合指标: AR, IR, Sharpe, WinRate"""
        if self.ret.ndim == 2:
            daily_ret = np.nanmean(self.ret, axis=0)  # 等权组合
        else:
            daily_ret = self.ret
        
        daily_ret = daily_ret[~np.isnan(daily_ret)]
        n_days = len(daily_ret)
        
        if n_days < 10:
            return {'AR': 0, 'IR': 0, 'Sharpe': 0, 'WinRate': 0}
        
        cumulative = np.cumprod(1 + daily_ret)
        
        # AR: Annual Return
        ar = (cumulative[-1] ** (252 / n_days) - 1) if n_days > 0 else 0
        
        # Sharpe
        sharpe = np.sqrt(252) * np.mean(daily_ret) / (np.std(daily_ret) + 1e-8)
        
        # IR: Information Ratio (excess over benchmark)
        if self.bench_ret is not None:
            excess = daily_ret - self.bench_ret[:n_days]
            ir = np.sqrt(252) * np.mean(excess) / (np.std(excess) + 1e-8)
        else:
            ir = sharpe  # 无基准时 IR ≈ Sharpe
        
        # WinRate
        win_rate = np.mean(daily_ret > 0)
        
        return {
            'AR': float(ar),
            'IR': float(ir),
            'Sharpe': float(sharpe),
            'WinRate': float(win_rate),
        }
    
    def _risk_metrics(self) -> dict:
        """风险指标: MaxDD, Volatility"""
        if self.ret.ndim == 2:
            daily_ret = np.nanmean(self.ret, axis=0)
        else:
            daily_ret = self.ret
        
        daily_ret = daily_ret[~np.isnan(daily_ret)]
        if len(daily_ret) < 10:
            return {'MaxDD': 0, 'AnnualVol': 0}
        
        cumulative = np.cumprod(1 + daily_ret)
        peak = np.maximum.accumulate(cumulative)
        max_dd = np.min((cumulative - peak) / peak)
        ann_vol = np.std(daily_ret) * np.sqrt(252)
        
        return {
            'MaxDD': float(max_dd),
            'AnnualVol': float(ann_vol),
        }


def run_qlib_backtest(predictions_file: str, returns_file: str, 
                      output_file: str, benchmark: str = "SH000300"):
    """运行完整 Qlib 兼容回测"""
    
    logger.info(f"Loading predictions: {predictions_file}")
    pred = np.load(predictions_file, allow_pickle=True)
    
    logger.info(f"Loading returns: {returns_file}")
    ret = np.load(returns_file, allow_pickle=True)
    
    metrics = QlibMetrics(pred, ret)
    results = metrics.compute_all()
    
    # 添加元数据
    results.update({
        'timestamp': datetime.now().isoformat(),
        'benchmark': benchmark,
        'n_days': pred.shape[1] if pred.ndim == 2 else len(pred),
    })
    
    # 打印结果
    logger.info("=" * 50)
    logger.info("  Qlib-Compatible Backtest Results")
    logger.info("=" * 50)
    logger.info(f"  IC Mean:     {results['IC_mean']:+.4f}")
    logger.info(f"  ICIR:        {results['ICIR']:+.4f}")
    logger.info(f"  RankIC:      {results['RankIC']:+.4f}")
    logger.info(f"  AR:          {results['AR']:+.4f} ({results['AR']*100:+.2f}%)")
    logger.info(f"  IR:          {results['IR']:+.4f}")
    logger.info(f"  Sharpe:      {results['Sharpe']:+.4f}")
    logger.info(f"  MaxDD:       {results['MaxDD']:+.4f} ({results['MaxDD']*100:+.2f}%)")
    logger.info(f"  WinRate:     {results['WinRate']:+.4f} ({results['WinRate']*100:.1f}%)")
    logger.info(f"  AnnualVol:   {results['AnnualVol']:+.4f} ({results['AnnualVol']*100:.1f}%)")
    logger.info("=" * 50)
    
    # 保存 JSON
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved: {output_file}")
    
    return results


def run_kronos_backtest(tokenizer_path: str, predictor_path: str,
                        data_dir: str, output_dir: str):
    """使用 Kronos 模型运行回测并生成 Qlib 兼容报告"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch
    import pickle
    
    from config.model_configs import build_tokenizer_config, build_model_config
    from model.kronos_tokenizer import KronosTokenizer
    from model.kronos_model import Kronos
    from model.predictor import KronosPredictor
    
    logger.info("Loading Kronos models...")
    
    # Load tokenizer (unpack config dict)
    tokenizer_config = build_tokenizer_config('mini')
    tokenizer = KronosTokenizer(**tokenizer_config)
    ckpt_tok = torch.load(tokenizer_path, map_location='cpu')
    tok_sd = ckpt_tok.get('model_state_dict', ckpt_tok)
    tokenizer.load_state_dict(tok_sd)
    tokenizer.eval()
    
    # Load predictor model
    model_config = build_model_config('mini')
    model = Kronos(**model_config)
    ckpt_pred = torch.load(predictor_path, map_location='cpu')
    pred_sd = ckpt_pred.get('model_state_dict', ckpt_pred)
    model.load_state_dict(pred_sd)
    model.eval()
    
    # Create predictor
    predictor = KronosPredictor(model=model, tokenizer=tokenizer)
    
    logger.info(f"Loading data: {data_dir}")
    val_data = pickle.load(open(Path(data_dir) / "val_data.pkl", "rb"))
    
    # 提取特征和标签
    features = val_data['features']  # (N, T, 6)
    returns = val_data['returns']    # (N, T)
    
    # 生成预测
    logger.info("Generating predictions...")
    predictions = []
    for i in range(len(features)):
        feat = torch.FloatTensor(features[i]).unsqueeze(0)
        with torch.no_grad():
            pred = predictor.predict(feat).squeeze().numpy()
        predictions.append(pred)
    predictions = np.array(predictions)
    
    # 保存中间结果
    pred_file = Path(output_dir) / "predictions.npy"
    ret_file = Path(output_dir) / "returns.npy"
    np.save(pred_file, predictions)
    np.save(ret_file, returns)
    
    # 运行 Qlib 回测
    return run_qlib_backtest(
        predictions_file=str(pred_file),
        returns_file=str(ret_file),
        output_file=str(Path(output_dir) / "qlib_results.json"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qlib-Compatible Backtest")
    parser.add_argument("--predictions", help="预测文件 (.npy)")
    parser.add_argument("--returns", help="收益文件 (.npy)")
    parser.add_argument("--output", default="./outputs/qlib_results.json")
    
    # Kronos 模式
    parser.add_argument("--kronos", action="store_true", help="使用 Kronos 模型")
    parser.add_argument("--tokenizer_path", default="./outputs/tokenizer/best_model.pt")
    parser.add_argument("--predictor_path", default="./outputs/predictor/best_model.pt")
    parser.add_argument("--data_dir", default="./data/semiconductor_v2/processed")
    parser.add_argument("--output_dir", default="./outputs/qlib_metrics")
    
    args = parser.parse_args()
    
    if args.kronos:
        run_kronos_backtest(
            args.tokenizer_path, args.predictor_path,
            args.data_dir, args.output_dir
        )
    elif args.predictions and args.returns:
        run_qlib_backtest(args.predictions, args.returns, args.output)
    else:
        parser.print_help()
