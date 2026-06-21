#!/usr/bin/env python3
"""
COS 云备份工具 — 公司规章制度训练数据/模型备份
===============================================
独立脚本，不依赖医学项目。

用法:
  python scripts/cos_backup.py backup              # 全量备份 (数据+模型)
  python scripts/cos_backup.py backup --model-only  # 仅备份 LoRA 权重
  python scripts/cos_backup.py backup --data-only   # 仅备份数据
  python scripts/cos_backup.py backup --dry-run     # 预览，不实际上传
  python scripts/cos_backup.py list                 # 列出 COS 远程文件

COS 路径结构:
  LLMs-from-scratch/
    projects/
      company-regulations-training/
        data/
          train.txt               ← 训练数据
          val.txt                 ← 验证数据
        output/
          best_model/             ← 最佳 LoRA 权重
          final_model/            ← 最终 LoRA 权重
          checkpoint/             ← 断点检查点
          training_log.json       ← 训练日志

配置:
  环境变量 COS_SECRET_ID / COS_SECRET_KEY
  或从 medica-handbook/.env / doc-search/.env 读取
  桶名: ins-kq6zz7wo-1313469539
  区域: ap-guangzhou
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# ── 项目路径 ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── COS 配置 (复用同一桶，独立前缀) ─────────────────────────
COS_BUCKET = "ins-kq6zz7wo-1313469539"
COS_REGION = "ap-guangzhou"
COS_PREFIX = "LLMs-from-scratch/projects/company-regulations-training/"

# ── 备份清单 ───────────────────────────────────────────────

# 训练数据目录
DATA_FILES = [
    "data/train.txt",
    "data/val.txt",
]

# LoRA 权重目录（备份整个目录）
MODEL_DIRS = [
    "output_stage1/best_model",
    "output_stage1/final_model",
    "output_stage1/checkpoint",
    "output_inst_v2/best_model",
    "output_inst_v2/checkpoint",
    "output/best_model",
    "output/final_model",
    "output/checkpoint",
    "output_inst/best_model",
    "output_inst/checkpoint",
]

# 训练日志
TRAINING_LOGS = [
    "output/training_log.json",
    "output_stage1/training_log.json",
    "output_inst_v2/training_log.json",
    "train.log",
    "train_stage1.log",
    "train_inst_v2.log",
    "train_inst.log",
]


def get_cos_client():
    """初始化 COS 客户端，从环境变量或 .env 文件读取凭证"""
    secret_id = os.environ.get("COS_SECRET_ID", "")
    secret_key = os.environ.get("COS_SECRET_KEY", "")

    if not secret_id or not secret_key:
        # 尝试从已知 .env 文件读取
        env_candidates = [
            Path("/mnt/d/codes/medica-handbook/.env"),
            Path("/mnt/d/docs/doc-search/.env"),
            Path.home() / ".cos.env",
        ]
        for env_file in env_candidates:
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("COS_SECRET_ID="):
                        secret_id = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("COS_SECRET_KEY="):
                        secret_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if secret_id and secret_key:
                    break

    if not secret_id or not secret_key:
        print("错误: 未找到 COS_SECRET_ID / COS_SECRET_KEY")
        print("  请设置环境变量:")
        print("    export COS_SECRET_ID='your_id'")
        print("    export COS_SECRET_KEY='your_key'")
        print("  或在以下位置放置 .env 文件:")
        for p in env_candidates:
            print(f"    {p}")
        sys.exit(1)

    from qcloud_cos import CosConfig, CosS3Client
    config = CosConfig(Region=COS_REGION, SecretId=secret_id, SecretKey=secret_key)
    return CosS3Client(config)


def local_to_remote(local_path: Path) -> str:
    """本地路径 → COS 对象键"""
    rel = local_path.relative_to(PROJECT_ROOT)
    return COS_PREFIX + str(rel).replace("\\", "/")


def collect_backup_files(model_only: bool = False, data_only: bool = False) -> list[Path]:
    """收集需要备份的本地文件"""
    files: list[Path] = []

    if not model_only:
        # 训练数据
        for df in DATA_FILES:
            f = PROJECT_ROOT / df
            if f.exists():
                files.append(f)
                size_str = f"{f.stat().st_size/1024/1024:.1f}MB" if f.stat().st_size > 1024*1024 else f"{f.stat().st_size/1024:.0f}KB"
                print(f"   [数据] {df} ({size_str})")

        # 训练日志
        for log in TRAINING_LOGS:
            f = PROJECT_ROOT / log
            if f.exists():
                files.append(f)
                print(f"   [日志] {log}")

    if not data_only:
        # LoRA 权重目录
        for model_dir in MODEL_DIRS:
            d = PROJECT_ROOT / model_dir
            if d.exists():
                count = 0
                for f in d.rglob("*"):
                    if f.is_file():
                        files.append(f)
                        count += 1
                print(f"   [模型] {model_dir}/ ({count} files)")

    return files


def list_remote_with_meta(client) -> dict[str, int]:
    """列出 COS 远程文件，返回 {key: size} 映射"""
    remote: dict[str, int] = {}
    marker = ""
    while True:
        resp = client.list_objects(
            Bucket=COS_BUCKET, Prefix=COS_PREFIX, Marker=marker, MaxKeys=1000
        )
        contents = resp.get("Contents", [])
        if not contents:
            break
        for obj in contents:
            remote[obj["Key"]] = int(obj["Size"])
        if resp.get("IsTruncated") == "true":
            marker = resp.get("NextMarker", contents[-1]["Key"])
        else:
            break
    return remote


def upload_file(client, local_path: Path, key: str) -> bool:
    """上传单个文件到 COS"""
    size = local_path.stat().st_size
    if size > 1024 * 1024:
        size_str = f"{size/1024/1024:.1f}MB"
    else:
        size_str = f"{size/1024:.0f}KB"

    try:
        print(f"  {size_str:>10}  {key}", end="", flush=True)
        client.upload_file(Bucket=COS_BUCKET, Key=key, LocalFilePath=str(local_path))
        print(" OK")
        return True
    except Exception as e:
        print(f" FAIL: {e}")
        return False


def cmd_backup(model_only: bool = False, data_only: bool = False,
               incremental: bool = False, dry_run: bool = False):
    """执行备份"""
    client = get_cos_client() if not dry_run else None

    print("\n收集本地文件...")
    files = collect_backup_files(model_only, data_only)

    if not files:
        print("\n没有找到需要备份的文件。")
        print("提示: 先运行数据准备 (data_prep.py) 和训练 (train_qwen_lora.py)")
        return

    files.sort(key=lambda x: x.stat().st_size)
    total_size = sum(f.stat().st_size for f in files)

    # 增量模式
    remote: dict[str, int] = {}
    if incremental and not dry_run:
        print("\n获取远程文件清单...")
        remote = list_remote_with_meta(client)
        print(f"远程已有 {len(remote)} 个文件")

    print(f"\n{'='*60}")
    print(f"备份到: cos://{COS_BUCKET}/{COS_PREFIX}")
    print(f"本地文件: {len(files)} 个, 总大小: {total_size/1024/1024:.1f} MB")
    if incremental:
        print(f"模式: 增量 (大小一致则跳过)")
    if dry_run:
        print(f"模式: 预览 (不实际上传)")
    print(f"{'='*60}\n")

    ok = skip = fail = 0
    uploaded_bytes = 0
    start = time.time()

    for f in files:
        key = local_to_remote(f)
        local_size = f.stat().st_size

        # 增量跳过
        if incremental and key in remote and remote.get(key) == local_size:
            skip += 1
            continue

        if dry_run:
            size_str = f"{local_size/1024/1024:.1f}MB" if local_size > 1024*1024 else f"{local_size/1024:.0f}KB"
            tag = "[替换]" if key in remote else "[新增]"
            print(f"  [DRY] {size_str:>10}  {tag} {key}")
            ok += 1
            continue

        if upload_file(client, f, key):
            ok += 1
            uploaded_bytes += local_size
        else:
            fail += 1

    elapsed = time.time() - start
    speed = uploaded_bytes / elapsed / 1024 if elapsed > 0 else 0

    print(f"\n{'='*60}")
    parts = [f"OK: {ok}"]
    if skip:
        parts.append(f"SKIP: {skip}")
    if fail:
        parts.append(f"FAIL: {fail}")
    parts.append(f"{elapsed:.0f}s")
    if uploaded_bytes > 0:
        parts.append(f"{speed:.0f}KB/s")
    print(" | ".join(parts))
    print(f"{'='*60}")


def cmd_list():
    """列出 COS 远程文件"""
    client = get_cos_client()

    all_keys = []
    marker = ""
    while True:
        resp = client.list_objects(
            Bucket=COS_BUCKET, Prefix=COS_PREFIX, Marker=marker, MaxKeys=1000
        )
        contents = resp.get("Contents", [])
        if not contents:
            break
        for obj in contents:
            all_keys.append((obj["Key"], obj["Size"]))
        if resp.get("IsTruncated") == "true":
            marker = resp.get("NextMarker", contents[-1]["Key"])
        else:
            break

    total_size = sum(int(s) for _, s in all_keys)
    print(f"\nCOS: cos://{COS_BUCKET}/{COS_PREFIX}")
    print(f"文件数: {len(all_keys)}, 总大小: {total_size/1024/1024:.1f} MB\n")
    for key, size in sorted(all_keys):
        size_int = int(size)
        if size_int > 1024*1024:
            size_str = f"{size_int/1024/1024:.1f}MB"
        else:
            size_str = f"{size_int/1024:.0f}KB"
        print(f"  {size_str:>10}  {key}")


def cmd_download(data_only: bool = False, model_only: bool = False,
                 output_dir: str = None):
    """从 COS 下载备份数据"""
    client = get_cos_client()

    if output_dir:
        target = Path(output_dir)
    else:
        target = PROJECT_ROOT
    target.mkdir(parents=True, exist_ok=True)

    # 确定下载范围
    prefixes = []
    if model_only:
        prefixes = [COS_PREFIX + "output/"]
    elif data_only:
        prefixes = [COS_PREFIX + "data/", COS_PREFIX + "train.log"]
    else:
        prefixes = [COS_PREFIX]

    all_keys = []
    for prefix in prefixes:
        marker = ""
        while True:
            resp = client.list_objects(
                Bucket=COS_BUCKET, Prefix=prefix, Marker=marker, MaxKeys=1000
            )
            contents = resp.get("Contents", [])
            if contents:
                all_keys.extend(contents)
            if resp.get("IsTruncated") == "true":
                marker = resp.get("NextMarker", contents[-1]["Key"])
            else:
                break

    if not all_keys:
        print("\nCOS 上没有找到可下载的文件。")
        return

    total_size = sum(int(obj["Size"]) for obj in all_keys)
    print(f"\n{'='*60}")
    print(f"从 COS 下载: cos://{COS_BUCKET}/{COS_PREFIX}")
    print(f"文件数: {len(all_keys)}, 总大小: {total_size/1024/1024:.1f} MB")
    print(f"目标目录: {target}")
    print(f"{'='*60}\n")

    ok = skip = fail = 0
    downloaded_bytes = 0
    start = time.time()

    for obj in sorted(all_keys, key=lambda x: x["Key"]):
        key = obj["Key"]
        size = int(obj["Size"])
        # 去除 COS 前缀 → 本地相对路径
        rel_path = key[len(COS_PREFIX):] if key.startswith(COS_PREFIX) else key
        local_file = target / rel_path
        local_file.parent.mkdir(parents=True, exist_ok=True)

        # 跳过已存在且大小相同的文件
        if local_file.exists() and local_file.stat().st_size == size:
            skip += 1
            continue

        size_str = f"{size/1024/1024:.1f}MB" if size > 1024*1024 else f"{size/1024:.0f}KB"
        try:
            print(f"  {size_str:>10}  {rel_path}", end="", flush=True)
            client.download_file(
                Bucket=COS_BUCKET, Key=key, DestFilePath=str(local_file)
            )
            print(" OK")
            ok += 1
            downloaded_bytes += size
        except Exception as e:
            print(f" FAIL: {e}")
            fail += 1

    elapsed = time.time() - start
    speed = downloaded_bytes / elapsed / 1024 if elapsed > 0 else 0

    print(f"\n{'='*60}")
    parts = [f"OK: {ok}"]
    if skip:
        parts.append(f"SKIP: {skip}")
    if fail:
        parts.append(f"FAIL: {fail}")
    parts.append(f"{elapsed:.0f}s")
    if downloaded_bytes > 0:
        parts.append(f"{speed:.0f}KB/s")
    print(" | ".join(parts))
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="COS 云备份 — 公司规章制度训练数据/模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/cos_backup.py backup                # 全量备份
  python scripts/cos_backup.py backup --model-only   # 仅备份模型权重
  python scripts/cos_backup.py backup --data-only    # 仅备份数据
  python scripts/cos_backup.py backup --incremental   # 增量备份
  python scripts/cos_backup.py backup --dry-run       # 预览
  python scripts/cos_backup.py list                   # 查看远程文件
  python scripts/cos_backup.py download               # 全量下载
  python scripts/cos_backup.py download --data-only   # 仅下载数据
  python scripts/cos_backup.py download --model-only  # 仅下载模型
        """,
    )
    sub = parser.add_subparsers(dest="command")

    # backup
    bp = sub.add_parser("backup", help="备份到 COS")
    bp.add_argument("--model-only", action="store_true", help="仅备份 LoRA 权重")
    bp.add_argument("--data-only", action="store_true", help="仅备份数据")
    bp.add_argument("--incremental", action="store_true", help="增量备份")
    bp.add_argument("--dry-run", action="store_true", help="预览，不实际上传")

    # list
    sub.add_parser("list", help="列出 COS 远程文件")

    dp = sub.add_parser("download", help="从 COS 下载备份")
    dp.add_argument("--model-only", action="store_true", help="仅下载 LoRA 权重")
    dp.add_argument("--data-only", action="store_true", help="仅下载数据")
    dp.add_argument("--output", type=str, default=None, help="下载目标目录 (默认: 项目根)")

    args = parser.parse_args()

    if args.command == "backup":
        cmd_backup(
            model_only=args.model_only,
            data_only=args.data_only,
            incremental=args.incremental,
            dry_run=args.dry_run,
        )
    elif args.command == "list":
        cmd_list()
    elif args.command == "download":
        cmd_download(
            data_only=args.data_only,
            model_only=args.model_only,
            output_dir=args.output,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
