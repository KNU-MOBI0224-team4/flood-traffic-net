#!/usr/bin/env python3
"""Build full-period D7 train-ready arrays without duplicating source CSV files."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PERCENTILES = ["p99", "p97"]
FOLDS = [
    "fold_1_train2016_2020_val2021_test2022",
    "fold_2_train2016_2021_val2022_test2023",
    "fold_3_train2016_2022_val2023_test2024",
]

SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2, "unused": 3}
CODE_TO_SPLIT = {value: key for key, value in SPLIT_TO_CODE.items()}


def dynamic_features(speed_feature_column: str) -> list[str]:
    return [
        "rainfall_mm_1h",
        speed_feature_column,
        "total_flow_median",
        "avg_occupancy_median",
        f"{speed_feature_column}_delta_1h",
        "total_flow_median_delta_1h",
        "avg_occupancy_median_delta_1h",
    ]


def base_traffic_features(speed_feature_column: str) -> dict[str, str]:
    return {
        speed_feature_column: speed_feature_column,
        "total_flow_median": "total_flow_median",
        "avg_occupancy_median": "avg_occupancy_median",
    }


def delta_features(speed_feature_column: str) -> list[tuple[str, str]]:
    return [
        (speed_feature_column, f"{speed_feature_column}_delta_1h"),
        ("total_flow_median", "total_flow_median_delta_1h"),
        ("avg_occupancy_median", "avg_occupancy_median_delta_1h"),
    ]


def safe_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_timestamps(monthly_dirs: list[Path]) -> list[str]:
    timestamps: set[str] = set()
    for monthly_dir in monthly_dirs:
        for path in sorted(monthly_dir.glob("*.csv.gz")):
            with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if row:
                        timestamps.add(row[0])
    return sorted(timestamps)


def continuous_hour_mask(timestamps: list[str]) -> np.ndarray:
    out = np.zeros(len(timestamps), dtype=np.bool_)
    parsed = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]
    for idx in range(1, len(parsed)):
        out[idx] = (parsed[idx] - parsed[idx - 1]).total_seconds() == 3600
    return out


def load_node_ids(nodes_csv: Path) -> list[str]:
    return [row["segment_id"] for row in read_csv_rows(nodes_csv)]


def build_graph(source_traffic_dir: Path, out_dir: Path, node_ids: list[str]) -> dict[str, Any]:
    graph_dir = out_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    node_count = len(node_ids)
    adjacency = np.zeros((node_count, node_count), dtype=np.uint8)
    edge_rows: list[dict[str, Any]] = []
    directed_edge_rows: list[dict[str, Any]] = []

    edges_path = source_traffic_dir / "D7_active_giant_adjacency_edges.csv"
    for row in read_csv_rows(edges_path):
        a_id = row["segment_id_a"]
        b_id = row["segment_id_b"]
        if a_id not in node_index or b_id not in node_index:
            continue
        a_idx = node_index[a_id]
        b_idx = node_index[b_id]
        adjacency[a_idx, b_idx] = 1
        adjacency[b_idx, a_idx] = 1
        edge_rows.append(
            {
                "segment_id_a": a_id,
                "segment_id_b": b_id,
                "node_idx_a": a_idx,
                "node_idx_b": b_idx,
                "adjacency_type": row.get("adjacency_type", ""),
            }
        )
        directed_edge_rows.append(
            {
                "source_idx": a_idx,
                "target_idx": b_idx,
                "source_segment_id": a_id,
                "target_segment_id": b_id,
            }
        )
        directed_edge_rows.append(
            {
                "source_idx": b_idx,
                "target_idx": a_idx,
                "source_segment_id": b_id,
                "target_segment_id": a_id,
            }
        )

    np.save(graph_dir / "A.npy", adjacency)
    write_csv(
        graph_dir / "node_ids.csv",
        ["node_idx", "segment_id"],
        [{"node_idx": idx, "segment_id": node_id} for idx, node_id in enumerate(node_ids)],
    )
    write_csv(
        graph_dir / "edges.csv",
        ["segment_id_a", "segment_id_b", "node_idx_a", "node_idx_b", "adjacency_type"],
        edge_rows,
    )
    write_csv(
        graph_dir / "edge_index.csv",
        ["source_idx", "target_idx", "source_segment_id", "target_segment_id"],
        directed_edge_rows,
    )

    spec = {
        "node_count": node_count,
        "undirected_edge_count": len(edge_rows),
        "directed_edge_count": len(directed_edge_rows),
        "self_loops": False,
        "adjacency_file": "A.npy",
        "source_edges": str(edges_path),
    }
    write_json(graph_dir / "graph_spec.json", spec)
    return spec


def build_dynamic_features(
    traffic_monthly_dir: Path,
    rainfall_monthly_dir: Path,
    out_dir: Path,
    node_ids: list[str],
    speed_feature_column: str,
) -> tuple[list[str], dict[str, Any]]:
    feature_dir = out_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_names = dynamic_features(speed_feature_column)
    traffic_feature_sources = base_traffic_features(speed_feature_column)
    delta_feature_pairs = delta_features(speed_feature_column)

    timestamps = collect_timestamps([traffic_monthly_dir, rainfall_monthly_dir])
    timestamp_index = {timestamp: idx for idx, timestamp in enumerate(timestamps)}
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    time_count = len(timestamps)
    node_count = len(node_ids)
    feature_count = len(feature_names)

    x = np.full((time_count, node_count, feature_count), np.nan, dtype=np.float32)
    feature_to_idx = {feature: idx for idx, feature in enumerate(feature_names)}

    rainfall_idx = feature_to_idx["rainfall_mm_1h"]
    for path in sorted(rainfall_monthly_dir.glob("D7_active_giant_node_rainfall_*.csv.gz")):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t_idx = timestamp_index.get(row["timestamp"])
                n_idx = node_index.get(row["segment_id"])
                if t_idx is None or n_idx is None:
                    continue
                x[t_idx, n_idx, rainfall_idx] = safe_float(row.get("rainfall_mm_1h"))

    for path in sorted(traffic_monthly_dir.glob("D7_active_giant_node_hourly_*.csv.gz")):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            missing_columns = sorted(set(traffic_feature_sources.values()) - set(reader.fieldnames or []))
            if missing_columns:
                raise ValueError(f"Missing traffic columns {missing_columns}: {path}")
            for row in reader:
                t_idx = timestamp_index.get(row["timestamp"])
                n_idx = node_index.get(row["segment_id"])
                if t_idx is None or n_idx is None:
                    continue
                for feature_name, source_col in traffic_feature_sources.items():
                    x[t_idx, n_idx, feature_to_idx[feature_name]] = safe_float(row.get(source_col))

    continuous = continuous_hour_mask(timestamps)
    for base_feature, delta_feature in delta_feature_pairs:
        base_idx = feature_to_idx[base_feature]
        delta_idx = feature_to_idx[delta_feature]
        valid_t = continuous.copy()
        current = x[1:, :, base_idx]
        previous = x[:-1, :, base_idx]
        delta = current - previous
        finite = np.isfinite(current) & np.isfinite(previous)
        x[1:, :, delta_idx] = np.where(valid_t[1:, None] & finite, delta, np.nan)

    input_mask = np.isfinite(x).astype(np.uint8)
    np.save(feature_dir / "X_dynamic.npy", x)
    np.save(feature_dir / "input_mask.npy", input_mask)
    np.save(feature_dir / "continuous_prev_hour.npy", continuous.astype(np.uint8))

    write_csv(
        feature_dir / "timestamps.csv",
        ["t_idx", "timestamp"],
        [{"t_idx": idx, "timestamp": timestamp} for idx, timestamp in enumerate(timestamps)],
    )
    write_json(
        feature_dir / "feature_spec.json",
        {
            "shape": {"T": time_count, "N": node_count, "F_dynamic": feature_count},
            "dynamic_features": feature_names,
            "delta_definition": "value[t] - value[t-1]; delta is missing across non-1-hour time gaps",
            "missing_values": "NaN in X_dynamic; input_mask is 1 where finite",
            "speed_feature_column": speed_feature_column,
            "source_traffic_monthly_dir": str(traffic_monthly_dir),
            "source_rainfall_monthly_dir": str(rainfall_monthly_dir),
        },
    )

    finite_by_feature = {
        feature: int(input_mask[:, :, idx].sum())
        for idx, feature in enumerate(feature_names)
    }
    return timestamps, {
        "time_count": time_count,
        "node_count": node_count,
        "feature_count": feature_count,
        "finite_by_feature": finite_by_feature,
    }


def build_label_set(
    label_root: Path,
    out_dir: Path,
    timestamps: list[str],
    node_ids: list[str],
    percentile: str,
    fold: str,
) -> dict[str, Any]:
    target_dir = out_dir / "labels" / percentile / fold
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp_index = {timestamp: idx for idx, timestamp in enumerate(timestamps)}
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    time_count = len(timestamps)
    node_count = len(node_ids)

    y = np.zeros((time_count, node_count), dtype=np.uint8)
    z = np.zeros((time_count, node_count), dtype=np.uint8)
    z_mask = np.zeros((time_count, node_count), dtype=np.uint8)
    label_available = np.zeros((time_count, node_count), dtype=np.uint8)
    split_code = np.full((time_count,), SPLIT_TO_CODE["unused"], dtype=np.uint8)

    label_dir = label_root / percentile / fold / "labels_monthly"
    for path in sorted(label_dir.glob(f"D7_active_giant_labels_{percentile}_*.csv.gz")):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t_idx = timestamp_index.get(row["timestamp"])
                n_idx = node_index.get(row["segment_id"])
                if t_idx is None or n_idx is None:
                    continue
                split_code[t_idx] = SPLIT_TO_CODE.get(row.get("split", "unused"), SPLIT_TO_CODE["unused"])
                label_available[t_idx, n_idx] = 1 if row.get("label_available") == "1" else 0
                y[t_idx, n_idx] = 1 if row.get("y") == "1" else 0
                z[t_idx, n_idx] = 1 if row.get("z") == "1" else 0
                z_mask[t_idx, n_idx] = 1 if row.get("z_mask") == "1" else 0

    np.save(target_dir / "y.npy", y)
    np.save(target_dir / "z.npy", z)
    np.save(target_dir / "z_mask.npy", z_mask)
    np.save(target_dir / "label_available.npy", label_available)
    np.save(target_dir / "split_code.npy", split_code)

    static_src = label_root / percentile / fold / "static_node_features_z_history.csv"
    thresholds_src = label_root / percentile / fold / f"D7_active_giant_label_node_thresholds_{percentile}.csv"
    summary_src = label_root / percentile / fold / f"D7_active_giant_labels_{percentile}_summary.json"
    shutil.copy2(static_src, target_dir / "X_static.csv")
    shutil.copy2(thresholds_src, target_dir / "label_node_thresholds.csv")
    shutil.copy2(summary_src, target_dir / "label_summary.json")

    split_summary: dict[str, dict[str, int]] = {}
    for split_name, code in SPLIT_TO_CODE.items():
        t_mask = split_code == code
        split_summary[split_name] = {
            "timestamps": int(t_mask.sum()),
            "z_positive": int(z[t_mask].sum()),
            "z_mask": int(z_mask[t_mask].sum()),
            "y_positive": int(y[t_mask].sum()),
            "label_available": int(label_available[t_mask].sum()),
        }

    spec = {
        "percentile": percentile,
        "fold": fold,
        "shape": {"T": time_count, "N": node_count},
        "target": "z",
        "state_label": "y",
        "loss_mask": "z_mask",
        "split_code": SPLIT_TO_CODE,
        "split_summary": split_summary,
        "source_label_dir": str(label_dir),
        "static_features": ["node_z_history_count_norm"],
    }
    write_json(target_dir / "target_spec.json", spec)
    return spec


def build_labels(label_root: Path, out_dir: Path, timestamps: list[str], node_ids: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for percentile in PERCENTILES:
        summary[percentile] = {}
        for fold in FOLDS:
            print(f"[labels] building {percentile} {fold}", flush=True)
            summary[percentile][fold] = build_label_set(
                label_root=label_root,
                out_dir=out_dir,
                timestamps=timestamps,
                node_ids=node_ids,
                percentile=percentile,
                fold=fold,
            )
    summaries_dir = out_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "D7_rolling_3fold_label_summary.csv",
        "static_node_features_z_history_all_summary.json",
    ]:
        src = label_root / filename
        if src.exists():
            shutil.copy2(src, summaries_dir / filename)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data"
    default_project_root = repo_root.parent / "Flood_induced_road_paralysis"
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--project-root", type=Path, default=default_project_root)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=data_root / "train_ready" / "d7_active_giant_full",
    )
    parser.add_argument("--label-root", type=Path, default=None)
    parser.add_argument("--speed-feature-column", default="avg_speed_median")
    parser.add_argument("--dataset-name", default=None)
    args = parser.parse_args()

    processed = args.project_root / "data" / "processed"
    traffic_dir = processed / "d7_active_giant_traffic"
    traffic_monthly_dir = traffic_dir / "traffic_monthly"
    rainfall_monthly_dir = processed / "d7_active_giant_rainfall" / "rainfall_monthly"
    label_root = args.label_root or processed / "d7_active_giant_labels_rolling_3fold"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name or out_dir.name

    node_ids = load_node_ids(traffic_dir / "D7_active_giant_nodes.csv")
    print(f"[graph] nodes={len(node_ids)}", flush=True)
    graph_summary = build_graph(traffic_dir, out_dir, node_ids)

    print("[features] building full dynamic features", flush=True)
    timestamps, feature_summary = build_dynamic_features(
        traffic_monthly_dir=traffic_monthly_dir,
        rainfall_monthly_dir=rainfall_monthly_dir,
        out_dir=out_dir,
        node_ids=node_ids,
        speed_feature_column=args.speed_feature_column,
    )
    print(
        f"[features] T={feature_summary['time_count']} N={feature_summary['node_count']} "
        f"F={feature_summary['feature_count']}",
        flush=True,
    )

    label_summary = build_labels(label_root, out_dir, timestamps, node_ids)

    manifest = {
        "dataset": dataset_name,
        "period": {"start": timestamps[0], "end": timestamps[-1], "timestamp_count": len(timestamps)},
        "graph": graph_summary,
        "features": feature_summary,
        "labels": label_summary,
        "source_paths": {
            "traffic_dir": str(traffic_dir),
            "traffic_monthly_dir": str(traffic_monthly_dir),
            "rainfall_monthly_dir": str(rainfall_monthly_dir),
            "label_root": str(label_root),
        },
        "speed_feature_column": args.speed_feature_column,
        "notes": [
            "No source monthly CSV/gz files are duplicated in this full train-ready directory.",
            "Static features are stored per percentile/fold and are not concatenated into X_dynamic.",
            "Tabular baselines concatenate static features at training time only.",
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
