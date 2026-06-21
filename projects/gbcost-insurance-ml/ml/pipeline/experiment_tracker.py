"""Lightweight MLflow experiment tracking — drop-in wrapper for train.py.

Reference: claim-modelling-kedro project uses Kedro + MLflow for experiment management.
This module provides MLflow integration WITHOUT requiring Kedro (too heavy for this project).

Supports:
  - Auto-logging: parameters, metrics, model artifacts, feature importance
  - Local tracking (file://) — zero server setup
  - Remote tracking (MLflow server) — for team collaboration
  - Drop-in: wrap existing train.py with one line

Usage:
    # In train.py or any training script:
    from ml.pipeline.experiment_tracker import track_experiment

    with track_experiment("my_experiment", config) as tracker:
        model = train(...)
        tracker.log_metrics({"gini": 0.52, "mae": 3200})
        tracker.log_model(model, "l1a_amount")
        tracker.log_artifact("models/feature_schema.json")
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """MLflow-compatible experiment tracker with local-fallback.

    When MLflow is not installed, stores runs as JSON files in experiments/mlruns/.
    When MLflow is available, uses MLflow tracking server.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "experiments/mlruns")
        self.tags = tags or {}
        self.run_id: Optional[str] = None
        self._use_mlflow = False
        self._start_time: float = 0
        self._local_dir: Optional[Path] = None

        # Try MLflow
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._use_mlflow = True
            self._mlflow = mlflow
        except ImportError:
            logger.info("MLflow not installed — using local JSON tracking")
            self._local_dir = Path(self.tracking_uri) / experiment_name
            self._local_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> "ExperimentTracker":
        self._start_time = time.time()
        if self._use_mlflow:
            self._mlflow.start_run(run_name=f"{self.experiment_name}_{int(self._start_time)}")
            self.run_id = self._mlflow.active_run().info.run_id
            for k, v in self.tags.items():
                self._mlflow.set_tag(k, v)
            logger.info("MLflow run started: %s", self.run_id)
        else:
            self.run_id = f"run_{int(self._start_time)}"
            (self._local_dir / self.run_id).mkdir(parents=True, exist_ok=True)
            logger.info("Local run started: %s", self.run_id)
        return self

    def log_params(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)):
                if self._use_mlflow:
                    self._mlflow.log_param(k, v)
                else:
                    self._log_local("params", {k: v}, merge=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._use_mlflow:
            self._mlflow.log_metrics(metrics, step=step)
        else:
            self._log_local("metrics", metrics, merge=True)

    def log_model(self, model, artifact_path: str = "model") -> None:
        if self._use_mlflow:
            import lightgbm as lgb
            if isinstance(model, lgb.Booster):
                self._mlflow.lightgbm.log_model(model, artifact_path)
            else:
                self._mlflow.sklearn.log_model(model, artifact_path) if hasattr(model, 'predict') else None

    def log_artifact(self, local_path: str) -> None:
        if self._use_mlflow:
            self._mlflow.log_artifact(local_path)
        else:
            import shutil
            src = Path(local_path)
            if src.exists():
                dst = self._local_dir / self.run_id / src.name
                shutil.copy2(src, dst)

    def log_dict(self, data: Dict, filename: str) -> None:
        if self._use_mlflow:
            self._mlflow.log_dict(data, filename)
        else:
            path = self._local_dir / self.run_id / filename
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def log_feature_importance(self, importance: Dict[str, float], top_n: int = 20) -> None:
        top = dict(sorted(importance.items(), key=lambda x: -x[1])[:top_n])
        self.log_dict({"feature_importance": top}, "feature_importance.json")

    def end(self) -> None:
        elapsed = time.time() - self._start_time
        self.log_metrics({"total_time_sec": round(elapsed, 1)})

        if self._use_mlflow:
            self._mlflow.end_run()
        else:
            summary = {
                "run_id": self.run_id,
                "experiment": self.experiment_name,
                "duration_sec": round(elapsed, 1),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            (self._local_dir / self.run_id / "run_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Run %s ended (%.1fs)", self.run_id, elapsed)

    def _log_local(self, key: str, data: Dict, merge: bool = False) -> None:
        path = self._local_dir / self.run_id / f"{key}.json"
        existing = {}
        if merge and path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        existing.update(data)
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.end()


@contextmanager
def track_experiment(
    name: str,
    config: Optional[Dict] = None,
    tags: Optional[Dict[str, str]] = None,
) -> ExperimentTracker:
    """Drop-in context manager for experiment tracking.

    Usage:
        with track_experiment("my_run", config) as tracker:
            tracker.log_params(config["l1a"])
            model = train(...)
            tracker.log_metrics({"gini": 0.52})
            tracker.log_model(model)
    """
    tracker = ExperimentTracker(name, tags=tags)
    tracker.start()
    if config:
        flat = {}
        for section in ["l1a", "l1b", "split", "features"]:
            if section in config:
                for k, v in config[section].items():
                    if isinstance(v, (int, float, str, bool)):
                        flat[f"{section}.{k}"] = v
        tracker.log_params(flat)
    try:
        yield tracker
    finally:
        tracker.end()


def list_experiments(tracking_uri: str = "experiments/mlruns") -> List[Dict]:
    """List all tracked experiments and their runs."""
    path = Path(tracking_uri)
    experiments = []
    if path.exists():
        for exp_dir in sorted(path.iterdir()):
            if exp_dir.is_dir():
                runs = []
                for run_dir in sorted(exp_dir.iterdir()):
                    if run_dir.is_dir():
                        summary = run_dir / "run_summary.json"
                        if summary.exists():
                            runs.append(json.loads(summary.read_text(encoding="utf-8")))
                experiments.append({"name": exp_dir.name, "runs": runs, "n_runs": len(runs)})
    return experiments
