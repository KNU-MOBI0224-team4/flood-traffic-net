"""Sequence-window data loading for STGCN.

Mirrors the policy of tabular_data.load_fold_data but keeps the (time, node)
structure intact: each sample is a (seq_len, N, F) input window with a
(N,) target and (N,) z_mask at target_t = last_input_t + pred_horizon.

Normalization uses mean/std computed on train-split timestamps only and is
applied to all splits to avoid leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from flood_traffic.constants import SPLIT_TO_CODE
from flood_traffic.io_utils import load_adjacency, load_node_ids, load_static
from flood_traffic.metrics import build_adjacency_list
from flood_traffic.sampling import sample_train_timestamps
from flood_traffic.tabular_data import TabularSplit, build_split_rows, build_y_state_rows


@dataclass
class STGCNFoldData:
    train_dataset: "STGCNDataset"
    val_dataset: "STGCNDataset"
    test_dataset: "STGCNDataset"
    A_hat: np.ndarray
    val_split: TabularSplit
    test_split: TabularSplit
    val_y_state_t: np.ndarray
    val_y_state_n: np.ndarray
    test_y_state_t: np.ndarray
    test_y_state_n: np.ndarray
    adjacency: list[list[int]]
    continuous_prev_hour: np.ndarray
    feature_stats: dict[str, Any]
    train_summary: dict[str, Any]
    dataset_summary: dict[str, Any]


class STGCNDataset(Dataset):
    """One sample = (input window, target labels, target mask, target_t).

    Shapes per sample:
        x:   (seq_len, N, F) float32
        y:   (N,) float32
        m:   (N,) bool
    """

    def __init__(
        self,
        X: np.ndarray,
        z: np.ndarray,
        z_mask: np.ndarray,
        target_timestamps: np.ndarray,
        seq_len: int,
        pred_horizon: int,
    ) -> None:
        self.X = X
        self.z = z
        self.z_mask = z_mask
        self.target_timestamps = np.asarray(target_timestamps, dtype=np.int64)
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

    def __len__(self) -> int:
        return int(len(self.target_timestamps))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        target_t = int(self.target_timestamps[idx])
        last_input_t = target_t - self.pred_horizon
        start = last_input_t - self.seq_len + 1
        x = np.ascontiguousarray(self.X[start : last_input_t + 1])
        y = np.ascontiguousarray(self.z[target_t]).astype(np.float32)
        m = np.ascontiguousarray(self.z_mask[target_t]).astype(np.bool_)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m), target_t


def _filter_in_bounds(
    target_ts: np.ndarray, seq_len: int, pred_horizon: int, T: int
) -> np.ndarray:
    min_target = seq_len - 1 + pred_horizon
    max_target = T - 1
    keep = (target_ts >= min_target) & (target_ts <= max_target)
    return target_ts[keep].astype(np.int64)


def load_graph_fold_data(
    data_dir: Path,
    percentile: str,
    fold: str,
    seq_len: int,
    pred_horizon: int,
    positive_timestamp_ratio: float,
    seed: int,
) -> STGCNFoldData:
    X_dynamic = np.asarray(np.load(data_dir / "features/X_dynamic.npy"), dtype=np.float32)
    input_mask = np.load(data_dir / "features/input_mask.npy", mmap_mode="r")
    continuous_prev_hour = np.load(
        data_dir / "features/continuous_prev_hour.npy", mmap_mode="r"
    ).astype(bool)
    A_hat = load_adjacency(data_dir / "graph/A.npy")
    A_raw = np.load(data_dir / "graph/A.npy", mmap_mode="r")
    adjacency = build_adjacency_list(np.asarray(A_raw))
    node_ids = load_node_ids(data_dir / "graph/node_ids.csv")

    target_dir = data_dir / "labels" / percentile / fold
    y_state = np.load(target_dir / "y.npy", mmap_mode="r")
    z = np.asarray(np.load(target_dir / "z.npy"), dtype=np.float32)
    z_mask = np.asarray(np.load(target_dir / "z_mask.npy")).astype(np.bool_)
    split_code = np.asarray(np.load(target_dir / "split_code.npy"))
    static, _ = load_static(target_dir / "X_static.csv", node_ids)

    selected_train_ts, train_summary = sample_train_timestamps(
        split_code=split_code,
        z=z,
        z_mask=z_mask,
        positive_timestamp_ratio=positive_timestamp_ratio,
        seed=seed,
    )
    val_ts = np.flatnonzero(split_code == SPLIT_TO_CODE["val"])
    test_ts = np.flatnonzero(split_code == SPLIT_TO_CODE["test"])

    T, N, F = X_dynamic.shape
    train_target_ts = _filter_in_bounds(np.asarray(selected_train_ts), seq_len, pred_horizon, T)
    val_target_ts = _filter_in_bounds(val_ts, seq_len, pred_horizon, T)
    test_target_ts = _filter_in_bounds(test_ts, seq_len, pred_horizon, T)

    train_mask = split_code == SPLIT_TO_CODE["train"]
    X_train_view = X_dynamic[train_mask]
    mean = np.nanmean(X_train_view, axis=(0, 1), keepdims=True)
    std = np.nanstd(X_train_view, axis=(0, 1), keepdims=True)
    std_safe = np.where(std > 1e-6, std, 1.0)

    X_norm = (X_dynamic - mean) / std_safe
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    train_dataset = STGCNDataset(X_norm, z, z_mask, train_target_ts, seq_len, pred_horizon)
    val_dataset = STGCNDataset(X_norm, z, z_mask, val_target_ts, seq_len, pred_horizon)
    test_dataset = STGCNDataset(X_norm, z, z_mask, test_target_ts, seq_len, pred_horizon)

    val_split = build_split_rows(
        X_dynamic, input_mask, z, z_mask, static, val_target_ts.astype(np.int32)
    )
    test_split = build_split_rows(
        X_dynamic, input_mask, z, z_mask, static, test_target_ts.astype(np.int32)
    )
    _, val_y_state_t, val_y_state_n = build_y_state_rows(
        X_dynamic, input_mask, y_state, static, val_target_ts.astype(np.int32)
    )
    _, test_y_state_t, test_y_state_n = build_y_state_rows(
        X_dynamic, input_mask, y_state, static, test_target_ts.astype(np.int32)
    )

    feature_stats = {
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
        "computed_on": "train-split timestamps only",
    }
    dataset_summary = {
        "percentile": percentile,
        "fold": fold,
        "seq_len": int(seq_len),
        "pred_horizon": int(pred_horizon),
        "shapes": {
            "time_count": int(T),
            "node_count": int(N),
            "dynamic_feature_count": int(F),
        },
        "sampling": train_summary,
        "targets_after_window_filter": {
            "train": int(len(train_target_ts)),
            "val": int(len(val_target_ts)),
            "test": int(len(test_target_ts)),
        },
        "feature_stats": feature_stats,
        "static_features_used": False,
    }

    return STGCNFoldData(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        A_hat=A_hat,
        val_split=val_split,
        test_split=test_split,
        val_y_state_t=val_y_state_t,
        val_y_state_n=val_y_state_n,
        test_y_state_t=test_y_state_t,
        test_y_state_n=test_y_state_n,
        adjacency=adjacency,
        continuous_prev_hour=np.asarray(continuous_prev_hour),
        feature_stats=feature_stats,
        train_summary=train_summary,
        dataset_summary=dataset_summary,
    )
