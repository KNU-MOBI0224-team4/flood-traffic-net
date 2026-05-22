"""Small file IO helpers shared by training scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from flood_traffic.constants import STATIC_FEATURES


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

