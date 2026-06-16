#!/usr/bin/env python3
"""Kronos 模型 COS 云备份 — 备份训练完成的模型到腾讯云对象存储"""
import os, sys, json, time
from pathlib import Path
from datetime import datetime
from qcloud_cos import CosConfig, CosS3Client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COS_BUCKET = "ins-kq6zz7wo-1313469539"
COS_REGION = "ap-guangzhou"
COS_PREFIX = "LLMs-from-scratch/projects/kronos-stock-predictor"

# Credentials from environment
secret_id = os.environ.get("COS_SECRET_ID", "")
secret_key = os.environ.get("COS_SECRET_KEY", "")
if not secret_id or not secret_key:
    # Try reading from medica-handbook .env
    env_file = Path("/home/raw/medica-handbook/.env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("COS_SECRET_ID="):
                secret_id = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("COS_SECRET_KEY="):
                secret_key = line.split("=", 1)[1].strip().strip('"')

if not secret_id:
    print("ERROR: COS_SECRET_ID not set")
    sys.exit(1)

config = CosConfig(Region=COS_REGION, SecretId=secret_id, SecretKey=secret_key)
client = CosS3Client(config)

def upload_file(local_path: str, cos_key: str):
    local = Path(local_path)
    if not local.exists():
        print(f"  SKIP (not found): {local_path}")
        return
    size_mb = local.stat().st_size / (1024 * 1024)
    print(f"  Uploading: {local_path} ({size_mb:.1f}MB) → {cos_key}")
    client.upload_file(Bucket=COS_BUCKET, Key=cos_key, LocalFilePath=str(local))
    print(f"  ✓ Done")

def backup_model(name: str, model_dir: str):
    print(f"\n=== Backup: {name} ===")
    for fname in ["best_model.pt"]:
        local = f"{model_dir}/{fname}"
        cos_key = f"{COS_PREFIX}/outputs/{name}/{fname}"
        upload_file(local, cos_key)

    # Upload metadata
    meta = {
        "name": name,
        "backup_time": str(datetime.now()),
        "source_dir": model_dir,
    }
    meta_path = f"/tmp/kronos_backup_{name}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    upload_file(meta_path, f"{COS_PREFIX}/outputs/{name}/backup_meta.json")

# Backup P0 models
backup_model("tokenizer_p0", str(PROJECT_ROOT / "outputs/tokenizer_p0"))
backup_model("predictor_p0", str(PROJECT_ROOT / "outputs/predictor_p0"))

# Backup forecast
print(f"\n=== Backup: 7day forecast ===")
forecast_path = PROJECT_ROOT / "data/7day_forecast.json"
if forecast_path.exists():
    upload_file(str(forecast_path), f"{COS_PREFIX}/data/7day_forecast.json")

print(f"\n=== Backup Complete ===")
