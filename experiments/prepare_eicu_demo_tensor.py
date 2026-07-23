#!/usr/bin/env python3
"""
Prepare the PhysioNet eICU demo vitalPeriodic table as a generic tensor.

The output follows the generic format supported by ``load_from_local_tensor``:

    X.npy
    obs_mask.npy
    case_ids.npy
    feature_names.json
    client_ids_5hospital_clusters.npy
    metadata.json

Design:
    - first 24 hours after ICU unit admission
    - 5-minute bins, T=288
    - 6 vital features: HR, RR, SpO2, SysBP, DiaBP, MeanBP
    - cohort requires HR/RR/SpO2 coverage >= 50%
    - hospitals are greedily grouped into balanced hospital-cluster clients
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = "data/eicu_demo"
DEFAULT_OUTPUT_DIR = "data/eicu_demo_tensor_T288_D6"
FEATURES = [
    "heartrate",
    "respiration",
    "sao2",
    "systemicsystolic",
    "systemicdiastolic",
    "systemicmean",
]
FEATURE_NAMES = ["HR", "RR", "SpO2", "SysBP", "DiaBP", "MeanBP"]
CORE_FEATURES = ["heartrate", "respiration", "sao2"]
FEATURE_RANGES = {
    "heartrate": (20.0, 250.0),
    "respiration": (1.0, 80.0),
    "sao2": (50.0, 100.0),
    "systemicsystolic": (30.0, 300.0),
    "systemicdiastolic": (10.0, 200.0),
    "systemicmean": (20.0, 250.0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--core-min-coverage", type=float, default=0.5)
    p.add_argument("--seq-length", type=int, default=288)
    p.add_argument("--bin-minutes", type=int, default=5)
    return p.parse_args()


def clean_ranges(df: pd.DataFrame) -> dict[str, int]:
    report = {}
    for feature, (lo, hi) in FEATURE_RANGES.items():
        bad = df[feature].notna() & ((df[feature] < lo) | (df[feature] > hi))
        report[feature] = int(bad.sum())
        df.loc[bad, feature] = np.nan
    return report


def make_hospital_clusters(
    patient_rows: pd.DataFrame,
    num_clients: int,
) -> tuple[np.ndarray, dict]:
    counts = (
        patient_rows.groupby("hospitalid")
        .size()
        .sort_values(ascending=False)
    )
    cluster_hospitals = [[] for _ in range(num_clients)]
    cluster_sizes = [0 for _ in range(num_clients)]

    for hospital_id, n_stays in counts.items():
        k = int(np.argmin(cluster_sizes))
        cluster_hospitals[k].append(int(hospital_id))
        cluster_sizes[k] += int(n_stays)

    hospital_to_client = {
        hospital_id: client_id
        for client_id, hospitals in enumerate(cluster_hospitals)
        for hospital_id in hospitals
    }
    client_ids = patient_rows["hospitalid"].map(hospital_to_client).to_numpy(int)
    metadata = {
        "cluster_sizes": cluster_sizes,
        "cluster_num_hospitals": [len(h) for h in cluster_hospitals],
        "cluster_hospital_ids": cluster_hospitals,
    }
    return client_ids, metadata


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patient = pd.read_csv(input_dir / "patient.csv.gz")
    vital = pd.read_csv(input_dir / "vitalPeriodic.csv.gz")
    hospital = pd.read_csv(input_dir / "hospital.csv.gz")

    patient["unitdischargeoffset"] = pd.to_numeric(
        patient["unitdischargeoffset"], errors="coerce"
    )

    vital = vital[["patientunitstayid", "observationoffset", *FEATURES]].copy()
    cleaning_report = clean_ranges(vital)

    vital = vital[
        (vital["observationoffset"] >= 0)
        & (vital["observationoffset"] < args.seq_length * args.bin_minutes)
    ].copy()
    vital["time_bin"] = (
        vital["observationoffset"] // args.bin_minutes
    ).astype(int)

    long = vital.melt(
        id_vars=["patientunitstayid", "time_bin"],
        value_vars=FEATURES,
        var_name="feature",
        value_name="value",
    ).dropna(subset=["value"])
    long["feature_idx"] = long["feature"].map(
        {feature: i for i, feature in enumerate(FEATURES)}
    )

    grouped = (
        long.groupby(["patientunitstayid", "time_bin", "feature_idx"])["value"]
        .median()
        .reset_index()
    )

    eligible_patient = patient[patient["unitdischargeoffset"] >= 1440].copy()
    eligible_ids = set(eligible_patient["patientunitstayid"].tolist())
    grouped = grouped[grouped["patientunitstayid"].isin(eligible_ids)]

    coverage = (
        grouped.groupby(["patientunitstayid", "feature_idx"])["time_bin"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=range(len(FEATURES)), fill_value=0)
        / args.seq_length
    )
    core_indices = [FEATURES.index(feature) for feature in CORE_FEATURES]
    keep_ids = coverage.index[
        (coverage[core_indices] >= args.core_min_coverage).all(axis=1)
    ].to_numpy()

    patient_rows = (
        eligible_patient[eligible_patient["patientunitstayid"].isin(keep_ids)]
        .sort_values("patientunitstayid")
        .reset_index(drop=True)
    )
    patient_rows = patient_rows.merge(hospital, on="hospitalid", how="left")
    case_ids = patient_rows["patientunitstayid"].to_numpy(np.int64)
    case_to_row = {case_id: i for i, case_id in enumerate(case_ids)}

    X = np.full(
        (len(case_ids), args.seq_length, len(FEATURES)),
        np.nan,
        dtype=np.float32,
    )
    selected = grouped[grouped["patientunitstayid"].isin(case_to_row)]
    for row in selected.itertuples(index=False):
        i = case_to_row[int(row.patientunitstayid)]
        X[i, int(row.time_bin), int(row.feature_idx)] = np.float32(row.value)

    obs_mask = np.isfinite(X).astype(np.float32)
    client_ids, cluster_metadata = make_hospital_clusters(
        patient_rows, args.num_clients
    )

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "obs_mask.npy", obs_mask)
    np.save(output_dir / "case_ids.npy", case_ids)
    np.save(output_dir / f"client_ids_{args.num_clients}hospital_clusters.npy", client_ids)
    np.save(output_dir / "hospital_ids.npy", patient_rows["hospitalid"].to_numpy(np.int64))
    with (output_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    per_feature_missing = {
        name: float(1.0 - obs_mask[..., i].mean())
        for i, name in enumerate(FEATURE_NAMES)
    }
    metadata = {
        "source": "PhysioNet eICU Collaborative Research Database Demo 2.0.1",
        "input_dir": str(input_dir),
        "shape": list(X.shape),
        "feature_columns": FEATURES,
        "feature_names": FEATURE_NAMES,
        "target_feature_indices_recommended": [0, 1, 2],
        "target_feature_names_recommended": ["HR", "RR", "SpO2"],
        "seq_length": args.seq_length,
        "bin_minutes": args.bin_minutes,
        "window_minutes": args.seq_length * args.bin_minutes,
        "cohort_rule": (
            "unitdischargeoffset >= 1440 and HR/RR/SpO2 first-24h "
            f"coverage >= {args.core_min_coverage:.0%}"
        ),
        "num_clients": args.num_clients,
        "client_split": "greedy balanced hospital clusters",
        "client_cluster_metadata": cluster_metadata,
        "overall_missing_rate": float(1.0 - obs_mask.mean()),
        "per_feature_missing_rate": per_feature_missing,
        "cleaning_bounds": FEATURE_RANGES,
        "cleaned_value_counts": cleaning_report,
        "region_counts": patient_rows["region"].value_counts(dropna=False).to_dict(),
        "bed_category_counts": (
            patient_rows["numbedscategory"].value_counts(dropna=False).to_dict()
        ),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    with (output_dir / "T_info.txt").open("w", encoding="utf-8") as f:
        f.write(f"T={args.seq_length}, D={len(FEATURES)}, N={X.shape[0]}\n")

    print(f"Saved eICU demo tensor to: {output_dir.resolve()}")
    print(f"Shape: {X.shape}")
    print(f"Overall missing rate: {1.0 - obs_mask.mean():.1%}")
    print("Per-feature missing rates:")
    for name, rate in per_feature_missing.items():
        print(f"  {name:>8s}: {rate:.1%}")
    print("Client sizes:")
    for k in range(args.num_clients):
        print(f"  Client {k}: {(client_ids == k).sum()} stays")


if __name__ == "__main__":
    main()
