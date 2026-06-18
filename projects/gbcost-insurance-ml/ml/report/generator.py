"""执行分析报告生成器。

从训练结果生成完整的 Markdown 报告 + 结构化 JSON 数据文件。

输出文件:
    - reports/ml/execution_report.md     (人类可读 Markdown 报告)
    - reports/ml/execution_report.json    (下游使用的结构化数据)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _fmt_num(val: Any, suffix: str = "") -> str:
    """数值格式化（千分位 + K/M 缩写）。"""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if abs(val) >= 1_000_000:
            return f"{val / 1_000_000:.2f}M{suffix}"
        elif abs(val) >= 1_000:
            return f"{val / 1_000:.1f}K{suffix}"
        return f"{val:.2f}{suffix}"
    if isinstance(val, int):
        return f"{val:,}{suffix}"
    return str(val)


def _fmt_pct(val: Any) -> str:
    if val is None:
        return "N/A"
    return f"{val:.1f}%"


def _status_label(condition: bool) -> str:
    return "✅ 通过" if condition else "❌ 未达标"


def _mape_label(mape: float) -> str:
    if mape < 30:
        return "✅ 通过 (< 30%)"
    elif mape < 50:
        return "⚠️ 警告 (30-50%)"
    return "❌ 未达标 (> 50%)"


def _composite_assessment(mape: float, r2: float = 0, gini: float = 0, total_err: float = 0) -> str:
    """多指标综合评估（适用于右偏保险理赔数据）。

    MAPE 单独使用对右偏数据有误导性，
    需综合 R2、Gini、总量误差等多维度判断。
    """
    score = 0
    score += min(r2 * 40, 40)              # R2: 最高40分 (>0.8满分)
    score += min(gini * 30, 30)            # Gini: 最高30分 (>0.7满分)
    score += max(0, 20 - total_err * 0.5)  # 总量误差: 最高20分 (<20%满分)
    if mape < 50:
        score += 10                         # MAPE < 50% 时加10分

    if score >= 70:
        return f"模型优秀（综合评分 {score:.0f}/100）— 可投入生产使用"
    elif score >= 50:
        return f"模型中等（综合评分 {score:.0f}/100）— 可用但需关注不足之处"
    else:
        return f"模型待提升（综合评分 {score:.0f}/100）— 需进一步优化"


def generate_report(
    report_data: Dict[str, Any],
    output_path: Optional[str] = None,
) -> Path:
    """生成 Markdown + JSON 执行分析报告。

    Args:
        report_data: 训练报告数据字典（来自 train.py）
        output_path: 覆盖 Markdown 输出路径

    Returns:
        生成的 Markdown 文件路径
    """
    report_dir = Path("reports/ml")
    report_dir.mkdir(parents=True, exist_ok=True)

    md_path = Path(output_path) if output_path else report_dir / "execution_report.md"
    json_path = report_dir / "execution_report.json"

    val_metrics = report_data.get("val_metrics", {})
    baseline = report_data.get("baseline_metrics", {})
    train_summary = report_data.get("train_summary", {})
    top_features = report_data.get("top_features", [])
    backtest = report_data.get("backtest_results", [])

    mae = val_metrics.get("mae", 0)
    mape = val_metrics.get("mape", 0)
    gini = val_metrics.get("gini", 0)
    total_err = val_metrics.get("total_error_pct", 0)
    r2 = val_metrics.get("r2", 0)

    baseline_mae = baseline.get("baseline_mean_mae", 0)
    baseline_mape = baseline.get("baseline_mean_mape", 0)
    improvement_mae = (1 - mae / baseline_mae) * 100 if baseline_mae > 0 else 0
    improvement_mape = (1 - mape / baseline_mape) * 100 if baseline_mape > 0 else 0

    # ==================================================================
    # 构建 Markdown
    # ==================================================================
    lines: List[str] = []
    w = lines.append

    # --- 报告头 ---
    w("# GHB 理赔预测 — 执行分析报告")
    w("")
    w(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"> 模型版本: L1-A v1 (LightGBM)")
    w(f"> 数据源: `{report_data.get('csv_path', 'N/A')}`")
    w("")
    w("---")
    w("")

    # --- 执行摘要 ---
    w("## 一、执行摘要")
    w("")
    # 复合门禁: R2>0.5 & Gini>0.3 & 总量误差<40% & MAE改善>50%
    # 注意: total_err 为百分比刻度（26.41 表示 26.41%）
    composite_ok = (
        r2 > 0.5
        and gini > 0.3
        and total_err < 40
        and improvement_mae > 50
    )
    w(f"| 指标 | 结果 | 状态 |")
    w(f"|------|------|------|")
    w(f"| **MAPE** | {_fmt_pct(mape)} | {_mape_label(mape)} |")
    w(f"| **MAE** | {_fmt_num(mae)} 元 | — |")
    w(f"| **R2** | {r2:.4f} | {'✅ 通过 (> 0.5)' if r2 > 0.5 else '⚠️ 警告'} |")
    w(f"| **Gini系数** | {gini:.4f} | {'✅ 通过 (> 0.3)' if gini > 0.3 else '⚠️ 警告'} |")
    w(f"| **总量误差** | {_fmt_pct(total_err)} | {'✅ 通过 (< 10%)' if total_err < 10 else '⚠️ 警告' if total_err < 30 else '❌ 未达标'} |")
    w(f"| **较基线MAE改善** | {_fmt_pct(improvement_mae)} | — |")
    w(f"| **较基线MAPE改善** | {_fmt_pct(improvement_mape)} | — |")
    w(f"| **Phase 1 门禁** | R2>0.5 & Gini>0.3 & 误差<40% & 改善>50% | **{_status_label(composite_ok)}** |")
    w("")
    w(f"**综合评估**: {_composite_assessment(mape, r2, gini, total_err)}")
    w("")
    w("> **注**: MAPE 对右偏理赔数据不适用（大量小额案件拉高百分比误差），"
      "应结合 R2、Gini、总量误差等综合判断。")
    w("")

    # --- 数据概览 ---
    w("## 二、数据概览")
    w("")
    w(f"| 项目 | 值 |")
    w(f"|------|-----|")
    w(f"| 训练集行数 | {report_data.get('train_rows', 0):,} |")
    w(f"| 验证集行数 | {report_data.get('val_rows', 0):,} |")
    w(f"| 特征数 | {report_data.get('feature_count', 0)} |")
    w(f"| 分类特征数 | {report_data.get('categorical_feature_count', 0)} |")
    split_cfg = report_data.get("split_config", {})
    w(f"| 训练期 | {split_cfg.get('train_end', 'N/A')} |")
    w(f"| 验证期 | {split_cfg.get('val_end', 'N/A')} |")
    w(f"| 测试期截止 | {split_cfg.get('test_end', 'N/A')} |")
    w("")

    # --- 模型配置 ---
    w("## 三、模型配置")
    w("")
    model_params = report_data.get("model_params", {})
    w("```python")
    for k, v in model_params.items():
        w(f"  {k}: {v}")
    w("```")
    w("")

    # --- 训练结果 ---
    w("## 四、训练结果")
    w("")
    w(f"| 指标 | 值 |")
    w(f"|------|-----|")
    w(f"| 最佳迭代轮次 | {train_summary.get('best_iteration', 'N/A')} |")
    w(f"| 训练耗时 | {train_summary.get('train_time_sec', 'N/A')}秒 |")
    train_l1 = train_summary.get("train_l1")
    train_rmse = train_summary.get("train_rmse")
    val_l1 = train_summary.get("val_l1")
    val_rmse = train_summary.get("val_rmse")
    if train_l1 is not None:
        w(f"| 训练集 MAE | {_fmt_num(train_l1)} 元 |")
    if train_rmse is not None:
        w(f"| 训练集 RMSE | {_fmt_num(train_rmse)} 元 |")
    if val_l1 is not None:
        w(f"| 验证集 MAE (early stopping) | {_fmt_num(val_l1)} 元 |")
    if val_rmse is not None:
        w(f"| 验证集 RMSE (early stopping) | {_fmt_num(val_rmse)} 元 |")
    w("")

    # --- 验证集评估指标 ---
    w("## 五、验证集评估指标")
    w("")
    w("### 5.1 核心指标")
    w("")
    w(f"| 指标 | ML模型 | 基线(均值预测) | 改善幅度 |")
    w(f"|------|--------|-----------|------|")
    w(f"| MAE | {_fmt_num(mae)} 元 | {_fmt_num(baseline_mae)} 元 | {_fmt_pct(improvement_mae)} |")
    w(f"| MAPE | {_fmt_pct(mape)} | {_fmt_pct(baseline_mape)} | {_fmt_pct(improvement_mape)} |")
    w(f"| RMSE | {_fmt_num(val_metrics.get('rmse', 0))} 元 | — | — |")
    w(f"| R2 | {r2:.4f} | — | — |")
    w(f"| Gini | {gini:.4f} | — | — |")
    w("")

    w("### 5.2 保险行业指标")
    w("")
    w(f"| 指标 | 值 | 说明 |")
    w(f"|------|-----|------|")
    w(f"| 真实总额 | {_fmt_num(val_metrics.get('total_true', 0))} 元 | 验证集实际赔付 |")
    w(f"| 预测总额 | {_fmt_num(val_metrics.get('total_pred', 0))} 元 | 模型预测赔付 |")
    w(f"| 总量误差 | {_fmt_pct(total_err)} | (预测-真实)/真实 |")
    if "large_auc" in val_metrics:
        w(f"| 大额案件 AUC | {val_metrics.get('large_auc', 0):.4f} | 排序判别能力 |")
    if "large_aucpr" in val_metrics:
        w(f"| 大额案件 AUC-PR | {val_metrics.get('large_aucpr', 0):.4f} | 不平衡场景精确率 |")
    if "large_recall" in val_metrics:
        w(f"| 大额案件 Recall | {_fmt_pct(val_metrics.get('large_recall', 0) * 100)} | 大额案件命中率 |")
    if "loss_ratio_true" in val_metrics:
        w(f"| 赔付率(真实) | {val_metrics.get('loss_ratio_true', 0):.4f} | — |")
        w(f"| 赔付率(预测) | {val_metrics.get('loss_ratio_pred', 0):.4f} | — |")
        w(f"| 准备金偏差 | {_fmt_num(val_metrics.get('reserve_impact', 0))} 元 | 赔付率误差×保费 |")
    w("")

    # --- 特征重要性 ---
    w("## 六、特征重要性 (Top 20)")
    w("")
    w(f"| 排名 | 特征 | 重要性 (gain) | 占比 |")
    w(f"|------|------|-------------|------|")
    total_imp = sum(f["importance"] for f in top_features) if top_features else 1
    for i, feat in enumerate(top_features[:20], 1):
        name = feat["feature"]
        imp = feat["importance"]
        share = imp / total_imp * 100 if total_imp > 0 else 0
        w(f"| {i} | `{name}` | {imp:,.0f} | {share:.1f}% |")
    w("")

    # --- Walk-Forward 回测 ---
    if backtest:
        w("## 七、Walk-Forward 回测")
        w("")
        w(f"| 轮次 | 测试月份 | 训练行数 | 测试行数 | MAE | MAPE | Gini | 总量误差 |")
        w(f"|------|---------|---------|---------|-----|------|------|---------|")
        for bt in backtest:
            w(f"| {bt.get('iteration', '')} | {bt.get('test_period', '')} "
              f"| {bt.get('train_rows', 0):,} | {bt.get('test_rows', 0):,} "
              f"| {_fmt_num(bt.get('mae', 0))} | {_fmt_pct(bt.get('mape', 0))} "
              f"| {bt.get('gini', 0):.4f} | {_fmt_pct(bt.get('total_error_pct', 0))} |")
        w("")

        import numpy as np
        bt_mae = np.mean([b.get("mae", 0) for b in backtest])
        bt_mape = np.mean([b.get("mape", 0) for b in backtest])
        bt_gini = np.mean([b.get("gini", 0) for b in backtest])
        bt_mae_std = np.std([b.get("mae", 0) for b in backtest])
        w(f"**回测均值**: MAE={_fmt_num(bt_mae)} (±{_fmt_num(bt_mae_std)}), "
          f"MAPE={_fmt_pct(bt_mape)}, Gini={bt_gini:.4f}")
        w("")

    # --- 结论与建议 ---
    w("## 八、结论与建议")
    w("")
    w("### 8.1 模型评估")
    w("")
    w(f"- **预测精度**: MAPE = {_fmt_pct(mape)} — {_composite_assessment(mape, r2, gini, total_err)}")
    w(f"- **排序能力**: Gini = {gini:.4f} — "
      f"{'强排序能力' if gini > 0.4 else '中等排序能力' if gini > 0.2 else '弱排序能力'}")
    w(f"- **拟合优度**: R2 = {r2:.4f} — "
      f"{'解释方差 > 80%，拟合优良' if r2 > 0.8 else '解释方差 50-80%，可接受' if r2 > 0.5 else '解释方差偏低'}")
    w(f"- **基线对比**: 较均值预测基线，MAE 改善 {_fmt_pct(improvement_mae)}")
    w(f"- **训练效率**: {train_summary.get('train_time_sec', 'N/A')}秒 — "
      f"{'快速' if train_summary.get('train_time_sec', 999) < 600 else '中等'}")
    w("")

    w("### 8.2 Phase 1 门禁判定")
    w("")
    if composite_ok:
        w(f"- **复合指标全部达标**: ✅ 通过 — 可进入 Phase 2")
        w(f"  - R2={r2:.4f} (>0.5), Gini={gini:.4f} (>0.3), "
          f"总量误差={_fmt_pct(total_err)} (<40%), MAE改善={_fmt_pct(improvement_mae)} (>50%)")
        w("- 建议后续: L1-B 大额案件识别 + 推理管线 + 保单级预测")
    else:
        fail_items = []
        if r2 <= 0.5:
            fail_items.append(f"R2={r2:.4f} (需>0.5)")
        if gini <= 0.3:
            fail_items.append(f"Gini={gini:.4f} (需>0.3)")
        if total_err >= 40:
            fail_items.append(f"总量误差={_fmt_pct(total_err)} (需<40%)")
        if improvement_mae <= 50:
            fail_items.append(f"MAE改善={_fmt_pct(improvement_mae)} (需>50%)")
        w(f"- **复合指标未全部达标**: {'; '.join(fail_items)}")
        w("- 建议优化方向:")
        w("  1. 增加交互特征 (group_code × age_bucket)")
        w("  2. 调整 Tweedie power 参数 (1.1-1.9)")
        w("  3. 增加 Optuna 调优 trials")
        w("  4. 检查数据质量问题 (异常值/缺失值)")
    w("")

    # --- 系统信息 ---
    w("## 九、系统信息")
    w("")
    w(f"| 项目 | 值 |")
    w(f"|------|-----|")
    import platform
    w(f"| 操作系统 | {platform.system()} {platform.release()} |")
    w(f"| Python | {platform.python_version()} |")
    try:
        import lightgbm
        w(f"| LightGBM | {lightgbm.__version__} |")
    except ImportError:
        pass
    try:
        import polars
        w(f"| Polars | {polars.__version__} |")
    except ImportError:
        pass
    w(f"| 报告生成时间 | {time.strftime('%Y-%m-%d %H:%M:%S')} |")
    w("")
    w("---")
    w("*本报告由 GHB ML 子系统自动生成*")

    # 写入 Markdown
    md_content = "\n".join(lines)
    md_path.write_text(md_content, encoding="utf-8")

    # ==================================================================
    # 构建结构化 JSON（键名保持英文，值中文）
    # ==================================================================
    json_data = {
        "report_type": "execution_analysis",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_version": "l1a_v1",
        "phase": "Phase 1 (L1-A)",

        "executive_summary": {
            "mape": mape,
            "mae": mae,
            "gini": gini,
            "total_error_pct": total_err,
            "improvement_over_baseline_mae_pct": improvement_mae,
            "improvement_over_baseline_mape_pct": improvement_mape,
            "r2": r2,
            "phase1_gate_passed": composite_ok,
            "assessment": _composite_assessment(mape, r2, gini, total_err),
        },

        "data_overview": {
            "csv_path": report_data.get("csv_path"),
            "train_rows": report_data.get("train_rows"),
            "val_rows": report_data.get("val_rows"),
            "feature_count": report_data.get("feature_count"),
            "categorical_feature_count": report_data.get("categorical_feature_count"),
            "split_config": split_cfg,
        },

        "model_config": model_params,

        "training_results": train_summary,

        "evaluation": {
            "model_metrics": val_metrics,
            "baseline_metrics": baseline,
            "improvement": {
                "mae_pct": improvement_mae,
                "mape_pct": improvement_mape,
            },
        },

        "feature_importance": top_features,

        "backtest": {
            "iterations": len(backtest),
            "results": backtest,
        } if backtest else None,
    }

    json_path.write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    logger.info("报告已生成:")
    logger.info("  Markdown: %s", md_path)
    logger.info("  JSON:     %s", json_path)

    return md_path
