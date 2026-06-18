"""Evaluation pipeline CLI entry point.

Usage:
    python -m ml.pipeline.evaluate --report-data reports/ml/training_report_data.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("ml.evaluate_cli")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate evaluation report from training data")
    parser.add_argument("--report-data", default="reports/ml/training_report_data.json",
                        help="Training report data JSON path")
    parser.add_argument("--output", default=None, help="Output Markdown path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    report_path = Path(args.report_data)
    if not report_path.exists():
        logger.error("Report data not found: %s", report_path)
        return 1

    report_data = json.loads(report_path.read_text(encoding="utf-8"))

    from ml.report.generator import generate_report
    md_path = generate_report(report_data, output_path=args.output)

    logger.info("Report generated: %s", md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
