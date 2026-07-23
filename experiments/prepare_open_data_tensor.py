#!/usr/bin/env python3
"""
Convert the small MIMIC-style open_data.csv file into the generic tensor format.

Input CSV columns:
    RecordID, Time, resp_rate, heart_rate, temperature, sbp

Output directory:
    X.npy
    obs_mask.npy
    case_ids.npy
    feature_names.json
    metadata.json

The tensor has shape (N, T, D), where N is the number of RecordID values, T is
the fixed hourly sequence length, and D is the number of vital-sign features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = "data/open_data.csv"
DEFAULT_OUTPUT = "data/open_data_tensor_T24"
FEATURES = ["resp_rate", "heart_rate", "temperature", "sbp"]

# Conservative physiological bounds. Values outside these ranges are treated
# as natural missingness rather than real observations.
CLEANING_BOUNDS = {
    "resp_rate": (1.0, 80.0),
    "heart_rate": (20.0, 250.0),
    "temperature": (30.0, 45.0),
    "sbp": (30.0, 260.0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", default=DEFAULT_INPUT)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--seq-length", type=int, default=24)
    p.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not mark physiologically implausible values as missing.",
    )
    return p.parse_args()


def build_tensor(df: pd.DataFrame, seq_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = []
    case_ids = []

    for rid, group in df.groupby("RecordID", sort=True):
        group = group.sort_values("Time").reset_index(drop=True)
        if len(group) < seq_length:
            continue
        if len(group) > seq_length:
            group = group.iloc[:seq_length].copy()

        values = group[FEATURES].to_numpy(dtype=np.float32)
        records.append(values)
        case_ids.append(int(rid))

    if not records:
        raise ValueError("No records with the requested sequence length were found.")

    X = np.stack(records).astype(np.float32)
    case_ids_arr = np.asarray(case_ids, dtype=np.int64)
    obs_mask = np.isfinite(X).astype(np.float32)
    return X, obs_mask, case_ids_arr


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, parse_dates=["Time"])
    missing_required = {"RecordID", "Time", *FEATURES} - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")

    df = df[["RecordID", "Time", *FEATURES]].copy()
    df["RecordID"] = df["RecordID"].astype("int64")

    cleaning_report = {}
    if not args.no_clean:
        for feature, (lo, hi) in CLEANING_BOUNDS.items():
            values = df[feature]
            bad = values.notna() & ((values < lo) | (values > hi))
            cleaning_report[feature] = int(bad.sum())
            df.loc[bad, feature] = np.nan

    X, obs_mask, case_ids = build_tensor(df, args.seq_length)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "obs_mask.npy", obs_mask)
    np.save(output_dir / "case_ids.npy", case_ids)
    with (output_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(FEATURES, f, indent=2)

    missing_rates = {
        feature: float(1.0 - obs_mask[..., i].mean())
        for i, feature in enumerate(FEATURES)
    }
    metadata = {
        "source_csv": str(input_csv),
        "shape": list(X.shape),
        "feature_names": FEATURES,
        "seq_length": args.seq_length,
        "n_records": int(X.shape[0]),
        "overall_missing_rate": float(1.0 - obs_mask.mean()),
        "per_feature_missing_rate": missing_rates,
        "cleaning_bounds": {} if args.no_clean else CLEANING_BOUNDS,
        "cleaned_value_counts": cleaning_report,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with (output_dir / "T_info.txt").open("w", encoding="utf-8") as f:
        f.write(f"T={args.seq_length}, D={len(FEATURES)}, N={X.shape[0]}\n")

    print(f"Saved open-data tensor to: {output_dir.resolve()}")
    print(f"Shape: {X.shape}")
    print(f"Overall missing rate: {1.0 - obs_mask.mean():.2%}")
    print("Per-feature missing rates:")
    for feature, rate in missing_rates.items():
        print(f"  {feature:>12s}: {rate:.2%}")
    if cleaning_report:
        print("Cleaned implausible values:")
        for feature, count in cleaning_report.items():
            print(f"  {feature:>12s}: {count}")


if __name__ == "__main__":
    main()
