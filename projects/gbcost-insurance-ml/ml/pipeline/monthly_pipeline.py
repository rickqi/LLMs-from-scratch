"""Monthly automated ML retraining and prediction pipeline.

Triggered after Doris data refresh. Runs: train → predict → evaluate → calibrate → backup.

Usage:
    python -m ml.pipeline.monthly_pipeline
    python -m ml.pipeline.monthly_pipeline --skip-tune
    python -m ml.pipeline.monthly_pipeline --config ml/config_ml.yaml --csv data/doris/c001_ghb_poicy_clm_duty_d.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("ml.monthly")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class StepResult:
    def __init__(self, name: str):
        self.name = name
        self.start = time.time()
        self.success = False
        self.error: str | None = None

    def finish(self, ok: bool, error: str | None = None):
        self.elapsed = time.time() - self.start
        self.success = ok
        self.error = error
        status = "OK" if ok else "FAIL"
        logger.info("[%s] %s (%.1fs)", status, self.name, self.elapsed)
        if error:
            logger.error("  Error: %s", error)


def run_cmd(cmd: list[str], step_name: str, timeout: int = 1800) -> StepResult:
    result = StepResult(step_name)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            result.finish(True)
        else:
            stderr_tail = "\n".join(proc.stderr.strip().split("\n")[-5:])
            result.finish(False, f"exit={proc.returncode}: {stderr_tail}")
        if proc.stdout.strip():
            logger.debug("[%s stdout]\n%s", step_name, proc.stdout.strip()[-500:])
    except subprocess.TimeoutExpired:
        result.finish(False, f"timeout after {timeout}s")
    except Exception as e:
        result.finish(False, str(e))
    return result


def run_monthly_pipeline(
    csv_path: str,
    config_path: str = "ml/config_ml.yaml",
    skip_tune: bool = True,
    skip_backtest: bool = False,
    exp_name: str | None = None,
) -> dict:
    results: dict[str, StepResult] = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if exp_name is None:
        exp_name = f"monthly_{timestamp}"

    steps: list[tuple[str, list[str]]] = []

    if not skip_tune:
        steps.append(("tune", [
            sys.executable, "-m", "ml.pipeline.tune",
            "--config", config_path,
            "--trials", "30",
        ]))

    train_cmd = [
        sys.executable, "-m", "ml.pipeline.train",
        "--config", config_path,
        "--mode", "full",
        "--exp-name", exp_name,
    ]
    if skip_backtest:
        train_cmd.append("--no-backtest")
    steps.append(("train", train_cmd))

    steps.append(("predict_all", [
        sys.executable, "-m", "ml.pipeline.predict",
        "--csv", csv_path,
        "--config", config_path,
        "--policy", "ALL",
    ]))

    steps.append(("evaluate", [
        sys.executable, "-m", "ml.pipeline.evaluate",
        "--config", config_path,
    ]))

    steps.append(("backup", [
        sys.executable, "-m", "ml.pipeline.cos_backup",
        "models", "predictions", "reports",
    ]))

    logger.info("Monthly pipeline starting: %s (%d steps)", exp_name, len(steps))

    for name, cmd in steps:
        logger.info("--- Step: %s ---", name)
        result = run_cmd(cmd, name)
        results[name] = result
        if not result.success:
            logger.warning("Step '%s' failed — continuing with remaining steps", name)

    summary = {
        "pipeline": exp_name,
        "timestamp": timestamp,
        "csv_path": csv_path,
        "config_path": config_path,
        "steps": {
            name: {"success": r.success, "elapsed": round(r.elapsed, 1), "error": r.error}
            for name, r in results.items()
        },
        "all_passed": all(r.success for r in results.values()),
    }

    report_path = Path(f"reports/ml/monthly_pipeline_{timestamp}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    passed = sum(1 for r in results.values() if r.success)
    logger.info("Pipeline complete: %d/%d steps passed. Report: %s",
                 passed, len(results), report_path)

    if not summary["all_passed"]:
        failed = [n for n, r in results.items() if not r.success]
        logger.warning("FAILED STEPS: %s", ", ".join(failed))

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monthly ML retraining pipeline")
    parser.add_argument("--config", default="ml/config_ml.yaml")
    parser.add_argument("--csv", default=None, help="Doris CSV path (default: from config)")
    parser.add_argument("--skip-tune", action="store_true", default=True, help="Skip Optuna tuning (default)")
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning before training")
    parser.add_argument("--no-backtest", action="store_true", help="Skip walk-forward backtest")
    parser.add_argument("--exp-name", default=None, help="Experiment name prefix")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    if args.csv is None:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        csv_path = cfg["data"]["csv_path"]
    else:
        csv_path = args.csv

    if not Path(csv_path).exists():
        logger.error("CSV not found: %s", csv_path)
        return 1

    do_tune = args.tune and not args.skip_tune

    summary = run_monthly_pipeline(
        csv_path=csv_path,
        config_path=args.config,
        skip_tune=not do_tune,
        skip_backtest=args.no_backtest,
        exp_name=args.exp_name,
    )

    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
