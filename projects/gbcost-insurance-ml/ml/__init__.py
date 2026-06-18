"""GHB Claim Prediction ML Subsystem.

Independent ML pipeline for insurance claim amount prediction.
Loosely coupled with the main GHB cost-control system via file system I/O.

Modules:
    - data: CSV loading (Polars), feature engineering, time-safe splitting
    - models: LightGBM amount predictor (L1-A)
    - evaluate: Insurance metrics, walk-forward backtest
    - pipeline: CLI entry points (train / predict / evaluate)
    - report: Execution analysis report generator (Markdown + JSON)
"""

__version__ = "1.0.0"
