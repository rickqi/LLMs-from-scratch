"""Walk-forward backtest runner script."""
import json, logging, pickle
import polars as pl
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

with open("ml/config_ml.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

schema = json.loads(open("models/feature_schema.json", encoding="utf-8").read())
feature_cols = schema["feature_cols"]
categorical_cols = schema["categorical_cols"]

from ml.data.loader import load_doris_csv
from ml.data.feature_store import build_features
from ml.evaluate.backtest import walk_forward_backtest

lf = load_doris_csv("data/doris/c001_ghb_poicy_clm_duty_d.csv")
with open("data/ml_cache/global_stats.pkl", "rb") as f:
    raw = pickle.load(f)
global_stats = {
    k: (pl.DataFrame(v) if isinstance(v, dict) else v)
    for k, v in raw.items()
}

feature_lf = build_features(
    lf, global_stats,
    categorical_cols=categorical_cols,
    log_transform=False,
)

results = walk_forward_backtest(
    feature_lf,
    {"l1a": config["l1a"]},
    feature_cols,
    categorical_cols,
    train_months=18,
    test_months=1,
    max_iterations=3,
)

with open("reports/ml/backtest_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

print("\n=== Backtest Summary ===")
for r in results:
    it = r["iteration"]
    tp = r["test_period"]
    mae = r["mae"]
    mape = r["mape"]
    gini = r["gini"]
    err = r["total_error_pct"]
    print(f"  [{it}] Test: {tp} | MAE={mae:.0f} | MAPE={mape:.1f}% | Gini={gini:.3f} | Err={err:.1f}%")
print("Done.")
