#!/usr/bin/env python3
"""
COS 云备份工具 — 将 LoRA 权重和指令微调数据备份到腾讯云对象存储。

用法:
  python scripts/cos_backup.py backup              # 全量备份
  python scripts/cos_backup.py backup --lora-only  # 仅备份 LoRA 权重
  python scripts/cos_backup.py backup --data-only  # 仅备份指令数据
  python scripts/cos_backup.py backup --dry-run    # 预览，不实际上传
  python scripts/cos_backup.py list                # 列出 COS 远程文件
  python scripts/cos_backup.py pre-ft-backup       # 指令微调前完整备份

COS 路径结构:
  LLMs-from-scratch/
    projects/
      chinese-medical-text-generation/
        output_full/              ← 阶段1 LoRA 权重
        output_inst_v1/           ← 阶段2 LoRA 权重 (如已存在)
        docs/
          med_qa_cases.json       ← QA 原始数据
          med_instruction_chatml.json  ← ChatML 训练数据
          med_instruction_alpaca.json  ← Alpaca 训练数据
          med_qa_report.md        ← 质量报告
        scripts/
          med_qa_generator.py     ← QA 生成脚本

配置:
  环境变量 COS_SECRET_ID / COS_SECRET_KEY (从 medica-handbook .env 读取)
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
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"

# 工作空间根目录 (用于构建 COS 路径)
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent  # /home/LLMs-from-scratch

# ── COS 配置 (复用 medica-handbook 的桶) ───────────────────
COS_BUCKET = "ins-kq6zz7wo-1313469539"
COS_REGION = "ap-guangzhou"
COS_PREFIX = "LLMs-from-scratch/projects/chinese-medical-text-generation/"

# ── 备份清单 ───────────────────────────────────────────────

# LoRA 权重目录（备份整个目录）
LORA_DIRS = [
    "output_full/best_model",
    "output_full/final_model",
    "output_full/checkpoint",
]

# 指令微调相关数据文件
INSTRUCTION_FILES = [
    "docs/med_qa_cases.json",
    "docs/med_instruction_chatml.json",
    "docs/med_instruction_alpaca.json",
    "docs/med_qa_report.md",
    "docs/instruction_ft_plan.md",
    "scripts/med_qa_generator.py",
]

# 训练日志
TRAINING_LOGS = [
    "train_full.log",
    "training_log.json",  # in output_full/
]


def get_cos_client():
    """初始化 COS 客户端"""
    secret_id = os.environ.get("COS_SECRET_ID", "")
    secret_key = os.environ.get("COS_SECRET_KEY", "")

    if not secret_id or not secret_key:
        # 尝试从 medica-handbook .env 读取
        env_files = [
            Path("/mnt/d/codes/medica-handbook/.env"),
            Path("/mnt/d/docs/doc-search/.env"),
        ]
        for env_file in env_files:
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("COS_SECRET_ID="):
                        secret_id = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("COS_SECRET_KEY="):
                        secret_key = line.split("=", 1)[1].strip().strip('"').strip("'")

    if not secret_id or not secret_key:
        print("❌ 未找到 COS_SECRET_ID / COS_SECRET_KEY")
        print("   请设置环境变量或在 medica-handbook/.env 中配置")
        sys.exit(1)

    from qcloud_cos import CosConfig, CosS3Client
    config = CosConfig(Region=COS_REGION, SecretId=secret_id, SecretKey=secret_key)
    return CosS3Client(config)


def local_to_remote(local_path: Path) -> str:
    """本地路径 → COS 对象键"""
    rel = local_path.relative_to(PROJECT_ROOT)
    return COS_PREFIX + str(rel).replace("\\", "/")


def collect_backup_files(lora_only: bool = False, data_only: bool = False) -> list[Path]:
    """收集需要备份的本地文件"""
    files: list[Path] = []

    if not data_only:
        # 收集 LoRA 权重目录下的所有文件
        for lora_dir in LORA_DIRS:
            d = PROJECT_ROOT / lora_dir
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        files.append(f)
                print(f"   📁 {lora_dir}/ ({sum(1 for _ in d.rglob('*') if _.is_file())} files)")

        # 训练日志
        for log in TRAINING_LOGS:
            f = PROJECT_ROOT / log
            if f.exists():
                files.append(f)

    if not lora_only:
        # 指令数据文件
        for inf in INSTRUCTION_FILES:
            f = PROJECT_ROOT / inf
            if f.exists():
                files.append(f)
                print(f"   📄 {inf} ({f.stat().st_size:,} bytes)")

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
    """上传单个文件到 COS（覆盖同 key 旧文件）"""
    size = local_path.stat().st_size
    if size > 1024 * 1024:
        size_str = f"{size/1024/1024:.1f}MB"
    else:
        size_str = f"{size/1024:.0f}KB"

    try:
        print(f"  {size_str:>10}  {key}", end="", flush=True)
        client.upload_file(Bucket=COS_BUCKET, Key=key, LocalFilePath=str(local_path))
        print(" ✅")
        return True
    except Exception as e:
        print(f" ❌ {e}")
        return False


def cmd_backup(lora_only: bool = False, data_only: bool = False,
               incremental: bool = False, dry_run: bool = False):
    """执行备份。
    
    - 默认模式: 覆盖上传所有本地文件（COS 同 key 自动替换旧版本）
    - 增量模式 (--incremental): 对比远程文件大小，大小一致则跳过
    """
    client = get_cos_client()

    files = collect_backup_files(lora_only, data_only)
    files.sort(key=lambda x: x.stat().st_size)

    total_size = sum(f.stat().st_size for f in files)

    # 增量模式: 获取远程文件清单
    remote: dict[str, int] = {}
    if incremental and not dry_run:
        print("获取远程文件清单...")
        remote = list_remote_with_meta(client)
        print(f"远程已有 {len(remote)} 个文件\n")

    print(f"\n{'='*60}")
    print(f"备份到: cos://{COS_BUCKET}/{COS_PREFIX}")
    print(f"本地文件: {len(files)} 个, 总大小: {total_size/1024/1024:.1f} MB")
    if incremental:
        print(f"模式: 增量 (文件大小一致则跳过)")
    print(f"{'='*60}\n")

    ok = skip = fail = 0
    uploaded_bytes = 0
    start = time.time()

    for f in files:
        key = local_to_remote(f)
        local_size = f.stat().st_size

        # 增量跳过: 远程存在且大小一致
        if incremental and key in remote and remote[key] == local_size:
            skip += 1
            continue

        if dry_run:
            action = "跳过" if (incremental and key in remote and remote.get(key) == local_size) else "上传"
            size_str = f"{local_size/1024/1024:.1f}MB" if local_size > 1024*1024 else f"{local_size/1024:.0f}KB"
            replacing = " [替换旧版]" if (key in remote and remote.get(key) != local_size) else ""
            print(f"  [DRY] {size_str:>10}  {key}{replacing}")
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
    parts = [f"✅ {ok} 上传"]
    if skip:
        parts.append(f"⏭️ {skip} 跳过")
    if fail:
        parts.append(f"❌ {fail} 失败")
    parts.append(f"{elapsed:.0f}s")
    if uploaded_bytes > 0:
        parts.append(f"{speed:.0f}KB/s")
    print(f"完成: {' | '.join(parts)}")
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
        print(f"  {int(size):>10,}  {key}")


def cmd_pre_ft_backup(dry_run: bool = False):
    """指令微调前完整备份"""
    print("=" * 60)
    print("  指令微调前完整备份")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    best_model = PROJECT_ROOT / "output_full" / "best_model"
    if not best_model.exists():
        print(f"\n⚠️  output_full/best_model 不存在")
        resp = input("   是否继续备份当前已有文件? [y/N]: ").strip().lower()
        if resp != 'y':
            print("   已取消")
            return

    print("\n📋 备份清单:\n")
    cmd_backup(lora_only=False, data_only=False, incremental=False, dry_run=dry_run)

    if not dry_run:
        train_log = PROJECT_ROOT / "train_full.log"
        last_lines = []
        if train_log.exists():
            last_lines = train_log.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]

        meta = {
            "backup_time": datetime.now().isoformat(),
            "backup_type": "pre-instruction-ft",
            "project": "LLMs-from-scratch/projects/chinese-medical-text-generation",
            "cos_bucket": COS_BUCKET,
            "cos_prefix": COS_PREFIX,
            "training_status": {"last_log_lines": last_lines},
        }

        meta_file = PROJECT_ROOT / "docs" / "pre_ft_backup_meta.json"
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n📝 备份元数据: {meta_file}")


def main():
    parser = argparse.ArgumentParser(
        description="COS 云备份工具 — 备份 LoRA 权重和指令数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/cos_backup.py pre-ft-backup           # 指令微调前完整备份
  python scripts/cos_backup.py pre-ft-backup --dry-run # 预览备份清单
  python scripts/cos_backup.py backup --lora-only      # 仅备份 LoRA 权重
  python scripts/cos_backup.py backup --data-only      # 仅备份指令数据
  python scripts/cos_backup.py list                    # 列出 COS 远程文件
        """,
    )
    sub = parser.add_subparsers(dest="command")

    # pre-ft-backup
    pre_ft = sub.add_parser("pre-ft-backup", help="指令微调前完整备份")
    pre_ft.add_argument("--dry-run", action="store_true", help="预览，不实际上传")

    # backup
    backup_p = sub.add_parser("backup", help="备份指定内容")
    backup_p.add_argument("--lora-only", action="store_true", help="仅备份 LoRA 权重")
    backup_p.add_argument("--data-only", action="store_true", help="仅备份指令数据")
    backup_p.add_argument("--incremental", action="store_true", help="增量备份，大小一致的跳过")
    backup_p.add_argument("--dry-run", action="store_true", help="预览，不实际上传")

    # list
    sub.add_parser("list", help="列出 COS 远程文件")

    args = parser.parse_args()

    if args.command == "pre-ft-backup":
        cmd_pre_ft_backup(dry_run=args.dry_run)
    elif args.command == "backup":
        cmd_backup(
            lora_only=args.lora_only,
            data_only=args.data_only,
            incremental=args.incremental,
            dry_run=args.dry_run,
        )
    elif args.command == "list":
        cmd_list()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
