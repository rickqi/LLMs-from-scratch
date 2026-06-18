"""COS cloud backup for ML model artifacts.

Bucket: ins-kq6zz7wo-1313469539 (ap-guangzhou)
Path:   LLMs-from-scratch/projects/gbcost-insurance-ml/

Usage:
    python -m ml.pipeline.cos_backup models predictions reports
    python -m ml.pipeline.cos_backup              # backup all
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("ml.cos_backup")

BUCKET = "ins-kq6zz7wo-1313469539"
REGION = "ap-guangzhou"
BASE_PREFIX = "LLMs-from-scratch/projects/gbcost-insurance-ml"


def get_client():
    from qcloud_cos import CosConfig, CosS3Client

    secret_id = os.environ.get("COS_SECRET_ID")
    secret_key = os.environ.get("COS_SECRET_KEY")
    if not secret_id or not secret_key:
        raise RuntimeError(
            "COS_SECRET_ID and COS_SECRET_KEY environment variables required. "
            "Set them in .env or export before running."
        )

    config = CosConfig(Region=REGION, SecretId=secret_id, SecretKey=secret_key)
    return CosS3Client(config)


def _upload_dir(client, local_dir: str, cos_subdir: str, pattern: str = "*") -> int:
    local_path = Path(local_dir)
    if not local_path.exists():
        logger.warning("Directory not found, skipping: %s", local_path)
        return 0

    count = 0
    for f in local_path.rglob(pattern):
        if not f.is_file():
            continue
        rel = f.relative_to(local_path)
        cos_key = f"{BASE_PREFIX}/{cos_subdir}/{rel}"
        client.upload_file(Bucket=BUCKET, Key=cos_key, LocalFilePath=str(f))
        count += 1
        logger.debug("  uploaded: %s", cos_key)

    return count


def backup_models(versions_dir: str = "models/versions", registry_file: str = "models/model_registry.json") -> int:
    client = get_client()
    count = 0
    count += _upload_dir(client, versions_dir, "models/versions")
    count += _upload_dir(client, Path(registry_file).parent, "models", pattern=Path(registry_file).name)
    logger.info("Models backup: %d files", count)
    return count


def backup_predictions(pred_dir: str = "predictions") -> int:
    client = get_client()
    count = _upload_dir(client, pred_dir, "predictions")
    logger.info("Predictions backup: %d files", count)
    return count


def backup_reports(report_dir: str = "reports/ml") -> int:
    client = get_client()
    count = _upload_dir(client, report_dir, "reports")
    logger.info("Reports backup: %d files", count)
    return count


def backup_all() -> dict[str, int]:
    results = {}
    for name, fn in [("models", backup_models), ("predictions", backup_predictions), ("reports", backup_reports)]:
        try:
            results[name] = fn()
        except Exception as e:
            logger.error("%s backup failed: %s", name, e)
            results[name] = -1
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    targets = sys.argv[1:] if len(sys.argv) > 1 else ["models", "predictions", "reports"]
    for target in targets:
        fn = {"models": backup_models, "predictions": backup_predictions, "reports": backup_reports}.get(target)
        if fn:
            logger.info("Backing up: %s ...", target)
            fn()
        else:
            logger.warning("Unknown target: %s (use: models, predictions, reports)", target)
    logger.info("Backup complete.")
