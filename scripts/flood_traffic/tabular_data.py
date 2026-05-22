"""Train-ready data loading and tabular row construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flood_traffic.constants import SPLIT_TO_CODE
from flood_traffic.io_utils import load_node_ids, load_static
from flood_traffic.metrics import build_adjacency_list
from flood_traffic.sampling import sample_train_timestamps


@dataclass
class TabularSplit:
    X: np.ndarray
    y: np.ndarray
    global_t: np.ndarray
    node_idx: np.ndarray


@dataclass
class FoldData:
    train: TabularSplit
    val: TabularSplit
    test: TabularSplit
    y_state_val_X: np.ndarray
    y_state_test_X: np.ndarray
    adjacency: list[list[int]]
    continuous_prev_hour: np.ndarray
    train_summary: dict[str, Any]
    dataset_summary: dict[str, Any]


def build_split_rows(
    X_dynamic: np.ndarray,
    input_mask: np.ndarray,
    z: np.ndarray,
    z_mask: np.ndarray,
    static: np.ndarray,
    timestamps: np.ndarray,
) -> TabularSplit:
    X_part = np.asarray(X_dynamic[timestamps], dtype=np.float32)
    input_ok = np.asarray(input_mask[timestamps].all(axis=2), dtype=bool)
    z_mask_part = np.asarray(z_mask[timestamps], dtype=bool)
    static_ok = np.isfinite(static).all(axis=1)
    valid = input_ok & z_mask_part & static_ok[None, :]

    time_count, node_count, feature_count = X_part.shape
    flat_valid = valid.reshape(-1)
    X_flat = X_part.reshape(-1, feature_count)[flat_valid]
    node_grid = np.broadcast_to(np.arange(node_count, dtype=np.int32)[None, :], (time_count, node_count))
    time_grid = np.broadcast_to(timestamps.astype(np.int32)[:, None], (time_count, node_count))
    node_flat = node_grid.reshape(-1)[flat_valid]
    time_flat = time_grid.reshape(-1)[flat_valid]
    static_flat = static[node_flat]
    X_out = np.hstack([X_flat, static_flat]).astype(np.float32)
    y_out = np.asarray(z[timestamps].reshape(-1)[flat_valid], dtype=np.uint8)
    return TabularSplit(X=X_out, y=y_out, global_t=time_flat, node_idx=node_flat)


def build_y_state_rows(
    X_dynamic: np.ndarray,
    input_mask: np.ndarray,
    y_state: np.ndarray,
    static: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_part = np.asarray(X_dynamic[timestamps], dtype=np.float32)
    input_ok = np.asarray(input_mask[timestamps].all(axis=2), dtype=bool)
    y_part = np.asarray(y_state[timestamps], dtype=bool)
    static_ok = np.isfinite(static).all(axis=1)
    valid = input_ok & y_part & static_ok[None, :]
    if not valid.any():
        return (
            np.empty((0, X_part.shape[2] + static.shape[1]), dtype=np.float32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )
    time_count, node_count, feature_count = X_part.shape
    flat_valid = valid.reshape(-1)
    X_flat = X_part.reshape(-1, feature_count)[flat_valid]
    node_grid = np.broadcast_to(np.arange(node_count, dtype=np.int32)[None, :], (time_count, node_count))
    time_grid = np.broadcast_to(timestamps.astype(np.int32)[:, None], (time_count, node_count))
    node_flat = node_grid.reshape(-1)[flat_valid]
    time_flat = time_grid.reshape(-1)[flat_valid]
    X_out = np.hstack([X_flat, static[node_flat]]).astype(np.float32)
    return X_out, time_flat, node_flat


def load_fold_data(
    data_dir: Path,
    percentile: str,
    fold: str,
    positive_timestamp_ratio: float,
    seed: int,
) -> FoldData:
    X_dynamic = np.load(data_dir / "features/X_dynamic.npy", mmap_mode="r")
    input_mask = np.load(data_dir / "features/input_mask.npy", mmap_mode="r")
    continuous_prev_hour = np.load(data_dir / "features/continuous_prev_hour.npy", mmap_mode="r").astype(bool)
    A = np.load(data_dir / "graph/A.npy", mmap_mode="r")
    adjacency = build_adjacency_list(np.asarray(A))
    node_ids = load_node_ids(data_dir / "graph/node_ids.csv")

    target_dir = data_dir / "labels" / percentile / fold
    y_state = np.load(target_dir / "y.npy", mmap_mode="r")
    z = np.load(target_dir / "z.npy", mmap_mode="r")
    z_mask = np.load(target_dir / "z_mask.npy", mmap_mode="r")
    split_code = np.load(target_dir / "split_code.npy", mmap_mode="r")
    static, static_features = load_static(target_dir / "X_static.csv", node_ids)

    selected_train_ts, train_summary = sample_train_timestamps(
        split_code=np.asarray(split_code),
        z=np.asarray(z),
        z_mask=np.asarray(z_mask),
        positive_timestamp_ratio=positive_timestamp_ratio,
        seed=seed,
    )
    val_ts = np.flatnonzero(split_code == SPLIT_TO_CODE["val"]).astype(np.int32)
    test_ts = np.flatnonzero(split_code == SPLIT_TO_CODE["test"]).astype(np.int32)

    train_split = build_split_rows(X_dynamic, input_mask, z, z_mask, static, selected_train_ts)
    val_split = build_split_rows(X_dynamic, input_mask, z, z_mask, static, val_ts)
    test_split = build_split_rows(X_dynamic, input_mask, z, z_mask, static, test_ts)
    y_state_val_X, _yval_t, _yval_n = build_y_state_rows(X_dynamic, input_mask, y_state, static, val_ts)
    y_state_test_X, _ytest_t, _ytest_n = build_y_state_rows(X_dynamic, input_mask, y_state, static, test_ts)

    dataset_summary = {
        "percentile": percentile,
        "fold": fold,
        "features": {
            "dynamic_count": int(X_dynamic.shape[2]),
            "static_features": static_features,
            "total_tabular_features": int(train_split.X.shape[1]),
        },
        "sampling": train_summary,
        "rows": {
            "train": {"n": int(len(train_split.y)), "positive": int(train_split.y.sum())},
            "val": {"n": int(len(val_split.y)), "positive": int(val_split.y.sum())},
            "test": {"n": int(len(test_split.y)), "positive": int(test_split.y.sum())},
            "y1_val": int(len(y_state_val_X)),
            "y1_test": int(len(y_state_test_X)),
        },
        "missing_policy": "drop rows unless z_mask==1 and all dynamic/static input features are finite",
    }

    return FoldData(
        train=train_split,
        val=val_split,
        test=test_split,
        y_state_val_X=y_state_val_X,
        y_state_test_X=y_state_test_X,
        adjacency=adjacency,
        continuous_prev_hour=continuous_prev_hour,
        train_summary=train_summary,
        dataset_summary=dataset_summary,
    )

