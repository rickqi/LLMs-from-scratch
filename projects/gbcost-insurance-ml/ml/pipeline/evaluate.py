"""Auto evaluation report — metrics, model comparison, quality alerts.

Generates a comprehensive Chinese report from training data, model registry,
and prediction summaries. Supports Markdown and JSON output.

Usage:
    python -m ml.pipeline.evaluate --output reports/ml/
    python -m ml.pipeline.evaluate --report-data reports/ml/training_report_data.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ml.evaluate_cli")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_training_report(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry(path: str | Path = "models/model_registry.json") -> dict:
    path = Path(path)
    if not path.exists():
        return {"versions": [], "latest": None}
    return json.loads(path.read_text(encoding="utf-8"))


def load_prediction_summaries(pred_dir: str | Path = "predictions") -> List[dict]:
    pred_dir = Path(pred_dir)
    summaries = []
    for f in sorted(pred_dir.glob("*_summary.json")):
        try:
            summaries.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return summaries


def generate_quality_alerts(report: dict, registry: dict) -> List[Dict[str, Any]]:
    """Generate quality alerts based on thresholds."""
    alerts = []
    metrics = report.get("val_metrics", {})

    if metrics.get("gini", 0) < 0.3:
        alerts.append({"level": "🔴", "msg": f"Gini={metrics['gini']:.3f} — 低于有效信号阈值 0.3", "action": "增加特征或扩大训练数据"})
    elif metrics.get("gini", 0) < 0.5:
        alerts.append({"level": "🟡", "msg": f"Gini={metrics['gini']:.3f} — 排序能力中等", "action": "尝试 Optuna 调优或增加交互特征"})

    if metrics.get("total_error_pct", 100) > 50:
        alerts.append({"level": "🔴", "msg": f"总量误差 {metrics['total_error_pct']:.1f}% — 系统性偏差严重", "action": "训练 GroupCalibrator 或检查目标分布"})
    elif metrics.get("total_error_pct", 100) > 30:
        alerts.append({"level": "🟡", "msg": f"总量误差 {metrics['total_error_pct']:.1f}% — 需要校正", "action": "应用校准器"})

    if metrics.get("large_recall", 0) < 0.3:
        alerts.append({"level": "🟡", "msg": f"大额召回 {metrics['large_recall']:.1%} — 大案识别不足", "action": "训练 L1-B 大额分类器"})

    if report.get("best_iteration", 0) >= report.get("n_estimators", 100) * 0.95:
        alerts.append({"level": "🟡", "msg": f"best_iter={report['best_iteration']} 接近 n_estimators 上限", "action": "增加 n_estimators 或调整 learning_rate"})

    versions = registry.get("versions", [])
    if len(versions) >= 3:
        recent = versions[-3:]
        ginis = [v["metrics"].get("gini", 0) for v in recent]
        if len(ginis) >= 2 and ginis[-1] < ginis[-2] * 0.95:
            alerts.append({"level": "🔴", "msg": "最近版本 Gini 下降 >5%", "action": "回滚到上一版本或检查数据漂移"})

    return alerts


def generate_evaluation_report(
    report_data: dict,
    registry: Optional[dict] = None,
    prediction_summaries: Optional[List[dict]] = None,
) -> str:
    """Generate comprehensive Chinese evaluation report."""
    if registry is None:
        registry = load_registry()
    if prediction_summaries is None:
        prediction_summaries = load_prediction_summaries()

    metrics = report_data.get("val_metrics", {})
    l1a = report_data.get("model_params", {})
    top_features = report_data.get("top_features", [])
    backtest = report_data.get("backtest_results", [])

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# ML 模型评估报告",
        f"> 生成时间: {now} | 模型版本: {registry.get('latest', 'N/A')}",
        "",
        "---",
        "",
        "## 一、模型概览",
        "",
        f"| 项目 | 值 |",
        f"|------|----|",
        f"| 训练行数 | {report_data.get('train_rows', 'N/A'):,} |",
        f"| 验证行数 | {report_data.get('val_rows', 'N/A'):,} |",
        f"| 特征数 | {report_data.get('feature_count', 'N/A')} |",
        f"| 分类特征数 | {report_data.get('categorical_feature_count', 'N/A')} |",
        f"| 训练时间 | {report_data.get('train_summary', {}).get('train_time_sec', 0):.0f}s |",
        f"| best_iteration | {report_data.get('train_summary', {}).get('best_iteration', 'N/A')} |",
        "",
        "## 二、核心指标",
        "",
        "| 指标 | 值 | 评级 | 说明 |",
        "|------|----|:--:|------|",
    ]

    gini = metrics.get("gini", 0)
    gini_ok = "🟢" if gini > 0.5 else ("🟡" if gini > 0.3 else "🔴")
    lines.append(f"| Gini | {gini:.4f} | {gini_ok} | 排序能力 |")

    r2 = metrics.get("r2", 0)
    r2_ok = "🟢" if r2 > 0.3 else ("🟡" if r2 > 0.1 else "🔴")
    lines.append(f"| R² | {r2:.4f} | {r2_ok} | 解释方差 |")

    mae = metrics.get("mae", 0)
    lines.append(f"| MAE | {mae:,.0f} 元 | — | 平均绝对误差 |")

    err = metrics.get("total_error_pct", 0)
    err_ok = "🟢" if err < 30 else ("🟡" if err < 50 else "🔴")
    lines.append(f"| 总量误差 | {err:.1f}% | {err_ok} | 系统性偏差 |")

    lauc = metrics.get("large_auc", 0)
    lines.append(f"| 大额 AUC | {lauc:.4f} | — | 大案区分 |")

    lrec = metrics.get("large_recall", 0)
    lines.append(f"| 大额召回 | {lrec:.1%} | — | 大案查全率 |")

    lines.extend([
        "",
        "## 三、Top 10 特征重要性",
        "",
        "| # | 特征 | 重要性 (gain) |",
        "|---|------|-------------:|",
    ])
    for i, feat in enumerate(top_features[:10], 1):
        lines.append(f"| {i} | {feat.get('feature', feat)} | {feat.get('importance', 0):,.0f} |")

    lines.extend([
        "",
        "## 四、版本历史",
        "",
        "| 版本 | 时间 | Gini | R² | MAPE | 特征数 |",
        "|------|------|-----:|----:|-----:|------:|",
    ])
    for v in registry.get("versions", [])[-10:]:
        m = v.get("metrics", {})
        lines.append(f"| v{v['version']} | {v.get('timestamp','')[:8]} | {m.get('gini',0):.4f} | {m.get('r2',0):.4f} | {m.get('mape',0):.1f}% | {m.get('feature_count',0)} |")

    if backtest:
        lines.extend([
            "",
            "## 五、回测结果",
            "",
            "| 窗口 | 测试期 | Gini | MAE | MAPE | 总量误差 |",
            "|------|--------|-----:|----:|-----:|--------:|",
        ])
        for bt in backtest:
            lines.append(f"| {bt.get('iteration','')} | {bt.get('test_period','')} | {bt.get('gini',0):.4f} | {bt.get('mae',0):,.0f} | {bt.get('mape',0):.1f}% | {bt.get('total_error_pct',0):.1f}% |")

    if prediction_summaries:
        lines.extend([
            "",
            "## 六、预测汇总",
            "",
            f"| 保单 | 案件数 | Gini | 总量误差 |",
            f"|------|------:|-----:|--------:|",
        ])
        for s in prediction_summaries[:20]:
            lines.append(f"| {s.get('policy_id','')} | {s.get('case_count',0):,} | {s.get('gini',0):.4f} | {s.get('total_error_pct',0):.1f}% |")

    alerts = generate_quality_alerts(report_data, registry)
    if alerts:
        lines.extend([
            "",
            "## 七、质量预警",
            "",
            "| 级别 | 问题 | 建议 |",
            "|:----:|------|------|",
        ])
        for a in alerts:
            lines.append(f"| {a['level']} | {a['msg']} | {a['action']} |")

    lines.extend([
        "",
        "---",
        f"*报告自动生成于 {now}*",
    ])

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate ML evaluation report")
    parser.add_argument("--report-data", default="reports/ml/training_report_data.json",
                        help="Training report data JSON")
    parser.add_argument("--registry", default="models/model_registry.json",
                        help="Model registry JSON")
    parser.add_argument("--pred-dir", default="predictions",
                        help="Prediction summaries directory")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    report_data = load_training_report(args.report_data)
    if not report_data:
        logger.error("No training report found at %s", args.report_data)
        return 1

    registry = load_registry(args.registry)
    pred_summaries = load_prediction_summaries(args.pred_dir)

    report_md = generate_evaluation_report(report_data, registry, pred_summaries)

    output_dir = Path(args.output or "reports/ml")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = output_dir / f"evaluation_report_{timestamp}.md"
    md_path.write_text(report_md, encoding="utf-8")

    json_path = output_dir / f"evaluation_report_{timestamp}.json"
    alerts = generate_quality_alerts(report_data, registry)
    json_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "metrics": report_data.get("val_metrics", {}),
        "alerts": alerts,
        "registry_summary": {
            "total_versions": len(registry.get("versions", [])),
            "latest": registry.get("latest"),
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Report: %s (%d bytes)", md_path, md_path.stat().st_size)
    logger.info("JSON:   %s", json_path)

    # Print summary
    logger.info("=" * 50)
    logger.info("Evaluation Summary:")
    metrics = report_data.get("val_metrics", {})
    logger.info("  Gini: %.4f | R²: %.4f | MAE: %.0f | Error: %.1f%%",
                 metrics.get("gini", 0), metrics.get("r2", 0),
                 metrics.get("mae", 0), metrics.get("total_error_pct", 0))
    if alerts:
        logger.info("  Alerts: %d", len(alerts))
        for a in alerts:
            logger.info("    %s %s", a["level"], a["msg"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
