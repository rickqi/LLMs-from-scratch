"""Batch label enrichment from agent_state.json files.

Processes Phase 2 analysis outputs and generates enriched training labels
for the ML pipeline. Creates a closed-loop: analysis → labels → better models.

Usage:
    python -m ml.pipeline.enrich_labels --input output/ --output data/ml_cache/
    python -m ml.pipeline.enrich_labels --input output/ --output data/ml_cache/ --policy GP123
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("ml.enrich")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract ML training labels from agent_state.json files")
    parser.add_argument("--input", required=True,
                        help="Directory containing agent_state.json files (e.g. output/)")
    parser.add_argument("--output", default="data/ml_cache",
                        help="Output directory for label parquet")
    parser.add_argument("--policy", default=None,
                        help="Process single policy directory (e.g. GP123_20260609_185651)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    from ml.data.label_enricher import batch_extract_labels

    input_dir = Path(args.input)
    if args.policy and (input_dir / args.policy).exists():
        # Single policy mode
        agent_file = input_dir / args.policy / "agent_state.json"
        logger.info("Processing single policy: %s", agent_file)
        import json as _json
        state = _json.loads(agent_file.read_text(encoding="utf-8"))
        from ml.data.label_enricher import extract_case_labels
        labels = extract_case_labels(state)
    else:
        pattern = f"{args.policy or '*'}/agent_state.json"
        labels = batch_extract_labels(input_dir, pattern=pattern)

    if len(labels) == 0:
        logger.warning("No labels extracted. Check input directory.")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_path = output_dir / "enriched_labels.parquet"
    labels.write_parquet(label_path)
    logger.info("Labels saved: %s (%d cases)", label_path, len(labels))

    # Statistics
    import polars as pl
    stats = {
        "total_cases": len(labels),
        "fwa_pct": float(labels["fwa_flag"].mean() * 100),
        "drg_pct": float(labels["drg_over_budget"].mean() * 100),
        "medical_pct": float(labels["medical_irrational"].mean() * 100),
        "avg_hosp_z": float(labels["hospital_z_score"].mean()),
        "label_distribution": {
            "fwa": int(labels["fwa_flag"].sum()),
            "drg_over": int(labels["drg_over_budget"].sum()),
            "medical": int(labels["medical_irrational"].sum()),
        },
    }

    stats_path = output_dir / "enriched_labels_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Stats saved: %s", stats_path)
    logger.info("Label distribution: fwa=%.1f%%, drg=%.1f%%, medical=%.1f%%",
                 stats["fwa_pct"], stats["drg_pct"], stats["medical_pct"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
