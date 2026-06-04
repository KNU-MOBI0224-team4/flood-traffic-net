"""Small file IO helpers shared by training scripts."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from flood_traffic.constants import STATIC_FEATURES


TIME_FEATURE_NAMES = [
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
    "sin_month",
    "cos_month",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_node_ids(path: Path) -> list[str]:
    return [row["segment_id"] for row in read_csv_rows(path)]


def load_timestamps(path: Path) -> list[str]:
    return [row["timestamp"] for row in read_csv_rows(path)]


def load_static(static_csv: Path, node_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    rows = read_csv_rows(static_csv)
    by_node = {row["segment_id"]: row for row in rows}
    values = np.full((len(node_ids), len(STATIC_FEATURES)), np.nan, dtype=np.float32)
    for node_idx, node_id in enumerate(node_ids):
        row = by_node.get(node_id)
        if row is None:
            continue
        for feature_idx, feature in enumerate(STATIC_FEATURES):
            try:
                values[node_idx, feature_idx] = float(row[feature])
            except Exception:
                values[node_idx, feature_idx] = np.nan
    return values, STATIC_FEATURES.copy()


def load_adjacency(path: Path) -> np.ndarray:
    """Load adjacency from .npy and return D^(-1/2) (A + I) D^(-1/2)."""
    A = np.asarray(np.load(path, mmap_mode="r")).astype(np.float32, copy=True)
    A += np.eye(A.shape[0], dtype=np.float32)
    d_inv_sqrt = 1.0 / np.sqrt(A.sum(axis=1) + 1e-8)
    return (A * d_inv_sqrt[None, :]) * d_inv_sqrt[:, None]


def compute_time_features(timestamps_csv: Path) -> np.ndarray:
    """Compute calendar/cyclical time features from timestamps.csv.

    Returns a (T, 6) float32 array with columns
        [sin_hour, cos_hour, sin_dow, cos_dow, sin_month, cos_month]
    All values are bounded in [-1, 1] by construction, so no further scaling
    is needed before concatenating with z-scored dynamic features.

    The cyclical sin/cos encoding ensures continuity across cycle boundaries
    (e.g., 23h↔0h, Sun↔Mon, Dec↔Jan) — a model receiving raw integer hour
    would not know that hour 23 and hour 0 are adjacent.
    """
    timestamps = load_timestamps(timestamps_csv)
    n = len(timestamps)
    out = np.zeros((n, len(TIME_FEATURE_NAMES)), dtype=np.float32)
    for i, ts in enumerate(timestamps):
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        h = dt.hour
        dow = dt.weekday()         # 0=Mon ... 6=Sun
        m = dt.month - 1           # 0..11
        out[i, 0] = np.sin(2.0 * np.pi * h / 24.0)
        out[i, 1] = np.cos(2.0 * np.pi * h / 24.0)
        out[i, 2] = np.sin(2.0 * np.pi * dow / 7.0)
        out[i, 3] = np.cos(2.0 * np.pi * dow / 7.0)
        out[i, 4] = np.sin(2.0 * np.pi * m / 12.0)
        out[i, 5] = np.cos(2.0 * np.pi * m / 12.0)
    return out


def save_test_predictions(
    path: Path,
    global_t: np.ndarray,
    node_idx: np.ndarray,
    y_true: np.ndarray,
    score: np.ndarray,
) -> None:
    """Save per-cell test predictions as a gzip CSV.

    Columns: global_t, node_idx, y_true, score. ``y_true`` is the onset target z
    for the evaluated cell. Used by the case-study export to align each model's
    predictions with ground truth. A ``.gz`` suffix is gzip-compressed by numpy.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack(
        [
            np.asarray(global_t, dtype=np.int64),
            np.asarray(node_idx, dtype=np.int64),
            np.asarray(y_true, dtype=np.int64),
            np.asarray(score, dtype=np.float64),
        ]
    )
    np.savetxt(
        path,
        arr,
        delimiter=",",
        header="global_t,node_idx,y_true,score",
        comments="",
        fmt=["%d", "%d", "%d", "%.8g"],
    )

