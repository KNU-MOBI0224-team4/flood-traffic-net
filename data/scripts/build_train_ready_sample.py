from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
import struct
import sys
import zipfile
from array import array
from datetime import datetime
from pathlib import Path
from typing import Any


DYNAMIC_FEATURES = [
    "rainfall_mm_1h",
    "avg_speed_median",
    "total_flow_median",
    "avg_occupancy_median",
    "avg_speed_median_delta_1h",
    "total_flow_median_delta_1h",
    "avg_occupancy_median_delta_1h",
]

TRAFFIC_FEATURE_TO_SOURCE = {
    "avg_speed_median": "avg_speed_median",
    "total_flow_median": "total_flow_median",
    "avg_occupancy_median": "avg_occupancy_median",
}

FOLDS = [
    "fold_1_train2016_2020_val2021_test2022",
    "fold_2_train2016_2021_val2022_test2023",
    "fold_3_train2016_2022_val2023_test2024",
]

PERCENTILES = ["p99", "p97"]


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


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def npy_payload(shape: tuple[int, ...], descr: str, data: bytes) -> bytes:
    header = {
        "descr": descr,
        "fortran_order": False,
        "shape": shape,
    }
    header_text = repr(header)
    header_text = header_text[:-1] + ", }"
    header_bytes = header_text.encode("latin1")
    preamble_len = 10
    pad_len = 16 - ((preamble_len + len(header_bytes) + 1) % 16)
    header_bytes = header_bytes + (b" " * pad_len) + b"\n"
    if len(header_bytes) >= 65536:
        raise ValueError("NPY v1 header is too large")
    return b"\x93NUMPY" + b"\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + data


def float32_bytes(values: list[float]) -> bytes:
    arr = array("f", values)
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tobytes()


def uint8_bytes(values: bytearray | bytes | list[int]) -> bytes:
    if isinstance(values, bytes):
        return values
    if isinstance(values, bytearray):
        return bytes(values)
    return bytes(values)


def write_npz(path: Path, arrays: dict[str, tuple[tuple[int, ...], str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, (shape, descr, data) in arrays.items():
            zf.writestr(f"{name}.npy", npy_payload(shape, descr, data))


def collect_timestamps_from_gzip(path: Path) -> list[str]:
    timestamps: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.add(row["timestamp"])
    return sorted(timestamps)


def is_one_hour(prev_ts: str, current_ts: str) -> bool:
    prev = datetime.strptime(prev_ts, "%Y-%m-%d %H:%M:%S")
    current = datetime.strptime(current_ts, "%Y-%m-%d %H:%M:%S")
    return (current - prev).total_seconds() == 3600


def flat_index(t_idx: int, n_idx: int, f_idx: int, node_count: int, feature_count: int) -> int:
    return ((t_idx * node_count + n_idx) * feature_count) + f_idx


def build_dynamic_features(
    source_dir: Path,
    out_dir: Path,
    node_ids: list[str],
) -> tuple[list[str], dict[str, Any]]:
    sample_root = source_dir / "samples/d7_active_giant"
    month_dirs = sorted([d for d in sample_root.iterdir() if d.is_dir()])

    all_timestamps: set[str] = set()
    for month_dir in month_dirs:
        rainfall_path = month_dir / "rainfall" / f"D7_active_giant_node_rainfall_{month_dir.name}.csv.gz"
        if rainfall_path.exists():
            all_timestamps.update(collect_timestamps_from_gzip(rainfall_path))
            
    timestamps = sorted(list(all_timestamps))
    timestamp_index = {timestamp: idx for idx, timestamp in enumerate(timestamps)}
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

    time_count = len(timestamps)
    node_count = len(node_ids)
    feature_count = len(DYNAMIC_FEATURES)
    total = time_count * node_count * feature_count
    values = [math.nan] * total
    mask = bytearray(total)

    def set_value(timestamp: str, segment_id: str, feature_name: str, value: float) -> None:
        t_idx = timestamp_index.get(timestamp)
        n_idx = node_index.get(segment_id)
        if t_idx is None or n_idx is None:
            return
        f_idx = DYNAMIC_FEATURES.index(feature_name)
        idx = flat_index(t_idx, n_idx, f_idx, node_count, feature_count)
        values[idx] = value
        if math.isfinite(value):
            mask[idx] = 1

    for month_dir in month_dirs:
        rainfall_path = month_dir / "rainfall" / f"D7_active_giant_node_rainfall_{month_dir.name}.csv.gz"
        traffic_path = month_dir / "traffic" / f"D7_active_giant_node_hourly_{month_dir.name}.csv.gz"

        if rainfall_path.exists():
            with gzip.open(rainfall_path, "rt", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    set_value(row["timestamp"], row["segment_id"], "rainfall_mm_1h", safe_float(row.get("rainfall_mm_1h")))

        if traffic_path.exists():
            with gzip.open(traffic_path, "rt", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for feature_name, source_col in TRAFFIC_FEATURE_TO_SOURCE.items():
                        set_value(row["timestamp"], row["segment_id"], feature_name, safe_float(row.get(source_col)))

    delta_pairs = [
        ("avg_speed_median", "avg_speed_median_delta_1h"),
        ("total_flow_median", "total_flow_median_delta_1h"),
        ("avg_occupancy_median", "avg_occupancy_median_delta_1h"),
    ]
    for t_idx in range(1, time_count):
        if not is_one_hour(timestamps[t_idx - 1], timestamps[t_idx]):
            continue
        for n_idx in range(node_count):
            for base_feature, delta_feature in delta_pairs:
                base_idx = DYNAMIC_FEATURES.index(base_feature)
                delta_idx = DYNAMIC_FEATURES.index(delta_feature)
                current_idx = flat_index(t_idx, n_idx, base_idx, node_count, feature_count)
                prev_idx = flat_index(t_idx - 1, n_idx, base_idx, node_count, feature_count)
                out_idx = flat_index(t_idx, n_idx, delta_idx, node_count, feature_count)
                current_value = values[current_idx]
                prev_value = values[prev_idx]
                if math.isfinite(current_value) and math.isfinite(prev_value):
                    values[out_idx] = current_value - prev_value
                    mask[out_idx] = 1

    write_npz(
        out_dir / "features/X_dynamic.npz",
        {
            "X_dynamic": ((time_count, node_count, feature_count), "<f4", float32_bytes(values)),
            "input_mask": ((time_count, node_count, feature_count), "|u1", uint8_bytes(mask)),
        },
    )

    write_csv_rows(
        out_dir / "features/timestamps.csv",
        ["t_idx", "timestamp"],
        [{"t_idx": idx, "timestamp": timestamp} for idx, timestamp in enumerate(timestamps)],
    )
    write_csv_rows(
        out_dir / "features/node_ids.csv",
        ["node_idx", "segment_id"],
        [{"node_idx": idx, "segment_id": node_id} for idx, node_id in enumerate(node_ids)],
    )
    write_json(
        out_dir / "features/feature_spec.json",
        {
            "dataset": "d7_active_giant_full",
            "shape": {"T": time_count, "N": node_count, "F_dynamic": feature_count},
            "dynamic_features": DYNAMIC_FEATURES,
            "primary_dynamic_inputs": [
                "rainfall_mm_1h",
                "avg_speed_median",
                "total_flow_median",
                "avg_occupancy_median",
            ],
            "traffic_instability_features": [
                "avg_speed_median_delta_1h",
                "total_flow_median_delta_1h",
                "avg_occupancy_median_delta_1h",
            ],
            "delta_definition": "value[t] - value[t-1], using only past/current information at anchor time t",
            "missing_values": "NaN in X_dynamic; availability is stored in input_mask",
        },
    )

    finite_by_feature = {}
    for f_idx, feature_name in enumerate(DYNAMIC_FEATURES):
        count = 0
        for t_idx in range(time_count):
            for n_idx in range(node_count):
                idx = flat_index(t_idx, n_idx, f_idx, node_count, feature_count)
                count += int(mask[idx] == 1)
        finite_by_feature[feature_name] = count

    return timestamps, {
        "time_count": time_count,
        "node_count": node_count,
        "feature_count": feature_count,
        "finite_by_feature": finite_by_feature,
    }


def build_graph(source_dir: Path, out_dir: Path, node_ids: list[str]) -> dict[str, Any]:
    graph_src = source_dir / "graph/d7_active_giant"
    graph_out = out_dir / "graph"
    graph_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(graph_src / "D7_active_giant_nodes.csv", graph_out / "nodes.csv")
    shutil.copy2(graph_src / "D7_active_giant_adjacency_edges.csv", graph_out / "edges.csv")
    shutil.copy2(graph_src / "D7_active_giant_adjacency_matrix.csv", graph_out / "adjacency_matrix.csv")

    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    node_count = len(node_ids)
    adjacency = bytearray(node_count * node_count)
    directed_edge_rows: list[dict[str, Any]] = []
    undirected_edge_count = 0

    for row in read_csv_rows(graph_src / "D7_active_giant_adjacency_edges.csv"):
        a_id = row["segment_id_a"]
        b_id = row["segment_id_b"]
        if a_id not in node_index or b_id not in node_index:
            continue
        a_idx = node_index[a_id]
        b_idx = node_index[b_id]
        adjacency[a_idx * node_count + b_idx] = 1
        adjacency[b_idx * node_count + a_idx] = 1
        undirected_edge_count += 1
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

    write_npz(
        graph_out / "A.npz",
        {"A": ((node_count, node_count), "|u1", uint8_bytes(adjacency))},
    )
    write_csv_rows(
        graph_out / "edge_index.csv",
        ["source_idx", "target_idx", "source_segment_id", "target_segment_id"],
        directed_edge_rows,
    )
    write_json(
        graph_out / "graph_spec.json",
        {
            "node_count": node_count,
            "undirected_edge_count": undirected_edge_count,
            "directed_edge_count": len(directed_edge_rows),
            "adjacency": "A[i,j] = 1 when road node i and road node j are connected",
            "self_loops": False,
        },
    )
    return {
        "node_count": node_count,
        "undirected_edge_count": undirected_edge_count,
        "directed_edge_count": len(directed_edge_rows),
    }


def build_labels(
    source_dir: Path,
    out_dir: Path,
    timestamps: list[str],
    node_ids: list[str],
) -> dict[str, Any]:
    timestamp_index = {timestamp: idx for idx, timestamp in enumerate(timestamps)}
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    time_count = len(timestamps)
    node_count = len(node_ids)
    label_summary: dict[str, Any] = {}

    sample_root = source_dir / "samples/d7_active_giant"
    month_dirs = sorted([d for d in sample_root.iterdir() if d.is_dir()])

    for percentile in PERCENTILES:
        label_summary[percentile] = {}
        for fold in FOLDS:
            y = bytearray(time_count * node_count)
            z = bytearray(time_count * node_count)
            z_mask = bytearray(time_count * node_count)
            label_available = bytearray(time_count * node_count)
            split_values: set[str] = set()

            for month_dir in month_dirs:
                label_path = (
                    month_dir
                    / "labels"
                    / percentile
                    / fold
                    / f"D7_active_giant_labels_{percentile}_{month_dir.name}.csv.gz"
                )

                if not label_path.exists():
                    continue

                with gzip.open(label_path, "rt", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        t_idx = timestamp_index.get(row["timestamp"])
                        n_idx = node_index.get(row["segment_id"])
                        if t_idx is None or n_idx is None:
                            continue
                        idx = t_idx * node_count + n_idx
                        split_values.add(row["split"])
                        label_available[idx] = 1 if row.get("label_available") == "1" else 0
                        y[idx] = 1 if row.get("y") == "1" else 0
                        z[idx] = 1 if row.get("z") == "1" else 0
                        z_mask[idx] = 1 if row.get("z_mask") == "1" else 0

            target_dir = out_dir / "labels" / percentile / fold
            write_npz(
                target_dir / "targets.npz",
                {
                    "y": ((time_count, node_count), "|u1", uint8_bytes(y)),
                    "z": ((time_count, node_count), "|u1", uint8_bytes(z)),
                    "z_mask": ((time_count, node_count), "|u1", uint8_bytes(z_mask)),
                    "label_available": ((time_count, node_count), "|u1", uint8_bytes(label_available)),
                },
            )
            
            static_history_path = source_dir / "static/d7_active_giant/z_history" / percentile / fold / "static_node_features_z_history.csv"
            if static_history_path.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(static_history_path, target_dir / "X_static.csv")

            threshold_path = source_dir / "metadata/d7_active_giant/label_thresholds" / percentile / fold / f"D7_active_giant_label_node_thresholds_{percentile}.csv"
            if threshold_path.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(threshold_path, target_dir / "label_node_thresholds.csv")
            
            summary_path = source_dir / "metadata/d7_active_giant/label_summaries" / percentile / fold / f"D7_active_giant_labels_{percentile}_summary.json"
            if summary_path.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(summary_path, target_dir / "label_summary.json")

            write_json(
                target_dir / "target_spec.json",
                {
                    "target": "z",
                    "state_label": "y",
                    "loss_mask": "z_mask",
                    "rainfall_seed_percentile": percentile,
                    "fold": fold,
                    "shape": {"T": time_count, "N": node_count},
                    "split_values_present_in_sample": sorted(split_values),
                    "notes": [
                        "Train/eval loss should be computed only where z_mask == 1.",
                        "Current y == 1 node-time cells are excluded from z_mask.",
                    ],
                },
            )
            label_summary[percentile][fold] = {
                "y_positive": int(sum(y)),
                "z_positive": int(sum(z)),
                "z_mask": int(sum(z_mask)),
                "label_available": int(sum(label_available)),
            }
    return label_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing data_sample/.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root
    source_dir = repo_root / "data_sample"
    out_dir = repo_root / "data_train" / "d7_active_giant_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = read_csv_rows(source_dir / "graph/d7_active_giant/D7_active_giant_nodes.csv")
    node_ids = [row["segment_id"] for row in nodes]

    graph_summary = build_graph(source_dir, out_dir, node_ids)
    timestamps, feature_summary = build_dynamic_features(source_dir, out_dir, node_ids)
    label_summary = build_labels(source_dir, out_dir, timestamps, node_ids)

    (out_dir / "summaries").mkdir(parents=True, exist_ok=True)
    
    summary_csv_src = source_dir / "summaries/d7_active_giant/D7_rolling_3fold_label_summary.csv"
    if summary_csv_src.exists():
        shutil.copy2(
            summary_csv_src,
            out_dir / "summaries/D7_rolling_3fold_label_summary.csv",
        )
        
    summary_json_src = source_dir / "summaries/d7_active_giant/static_node_features_z_history_all_summary.json"
    if summary_json_src.exists():
        shutil.copy2(
            summary_json_src,
            out_dir / "summaries/static_node_features_z_history_all_summary.json",
        )

    write_json(
        out_dir / "manifest.json",
        {
            "dataset": "d7_active_giant_full",
            "source": "data_sample",
            "output": "data_train/d7_active_giant_full",
            "graph": graph_summary,
            "features": feature_summary,
            "labels": label_summary,
            "array_files": {
                "graph": "graph/A.npz",
                "dynamic_features": "features/X_dynamic.npz",
                "labels": "labels/{p97,p99}/{fold}/targets.npz",
            },
        },
    )

    print(f"Wrote train-ready sample to {out_dir}")
    print(f"T={feature_summary['time_count']} N={feature_summary['node_count']} F={feature_summary['feature_count']}")


if __name__ == "__main__":
    main()