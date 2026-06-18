"""Calculate per-bucket and per-duty_code correction factors."""
import json
import polars as pl
import numpy as np
from pathlib import Path

df = pl.read_parquet("predictions/ALL_case_predictions.parquet")

# 1. Global correction
total_actual = df["actual_amount"].sum()
total_pred = df["predicted_amount"].sum()
global_factor = total_actual / total_pred
print(f"Global: actual={total_actual:,.0f}, pred={total_pred:,.0f}, factor={global_factor:.4f}")

# 2. Per-bucket correction
buckets = ["0-500", "500-2K", "2K-10K", "10K-50K", "50K+"]
bucket_factors = {}
print("\nPer-bucket:")
for b in buckets:
    mask = df["amount_bucket"] == b
    a = df.filter(mask)["actual_amount"].sum()
    p = df.filter(mask)["predicted_amount"].sum()
    f = a / p if p > 0 else 1.0
    n = mask.sum()
    bucket_factors[b] = {"factor": round(f, 4), "count": n}
    print(f"  {b}: n={n:>10,}, factor={f:.4f}")

# 3. Per-duty_code (top 15)
print("\nTop 15 duty_codes:")
duty_stats = df.group_by("duty_code").agg(
    pl.col("actual_amount").sum().alias("actual"),
    pl.col("predicted_amount").sum().alias("pred"),
    pl.len().alias("n"),
).with_columns(
    (pl.col("actual") / pl.col("pred")).alias("factor")
).sort("n", descending=True).head(15)

duty_factors = {}
for row in duty_stats.iter_rows(named=True):
    k = row["duty_code"]
    f = round(row["factor"], 4)
    duty_factors[k] = {"factor": f, "n": row["n"]}
    print(f"  {k}: n={row['n']:>10,}, factor={f:.4f}")

# 4. Save correction table
correction = {
    "global_factor": round(global_factor, 4),
    "bucket_factors": bucket_factors,
    "duty_factors": duty_factors,
    "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
    "note": "Multiply predicted_amount by factor for corrected estimate. Global=total_actual/total_pred.",
}
Path("models/bucket_correction.json").write_text(
    json.dumps(correction, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"\nSaved: models/bucket_correction.json")

# 5. Show corrected totals
print("\nCorrected estimates:")
for b in buckets:
    mask = df["amount_bucket"] == b
    raw_pred = df.filter(mask)["predicted_amount"].sum()
    actual = df.filter(mask)["actual_amount"].sum()
    f = bucket_factors[b]["factor"]
    corrected = raw_pred * f
    err_before = abs(raw_pred - actual) / actual * 100
    err_after = abs(corrected - actual) / actual * 100
    print(f"  {b}: raw={raw_pred:,.0f}, corrected={corrected:,.0f}, actual={actual:,.0f}, err {err_before:.0f}% -> {err_after:.0f}%")

raw_total = df["predicted_amount"].sum()
corrected_total = raw_total * global_factor
print(f"\nTotal: raw={raw_total:,.0f}, corrected={corrected_total:,.0f}, actual={total_actual:,.0f}")
