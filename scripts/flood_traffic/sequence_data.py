"""Train-ready data loading and GRU sequence construction."""

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
class GRUSplit:
    X: np.ndarray  
    y: np.ndarray
    global_t: np.ndarray
    node_idx: np.ndarray


@dataclass
class FoldData:
    train: GRUSplit
    val: GRUSplit
    test: GRUSplit
    y_state_val_X: np.ndarray
    y_state_test_X: np.ndarray
    adjacency: list[list[int]]
    continuous_prev_hour: np.ndarray
    train_summary: dict[str, Any]
    dataset_summary: dict[str, Any]


def _build_sequence_windows(
    X_dynamic: np.ndarray,
    input_mask: np.ndarray,
    select: np.ndarray,
    static: np.ndarray,
    timestamps: np.ndarray,
    seq_len: int,
    pred_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (num_rows, seq_len, F+static) input windows.

    For each target timestamp ``t`` in ``timestamps`` and node ``n`` where
    ``select[t, n]`` is set, the window covers
    ``X[t-pred_horizon-seq_len+1 ... t-pred_horizon]`` — i.e. the last input
    step is ``t - pred_horizon`` and the target is anchored at ``t``. This
    mirrors the STGCN window policy so the GRU baseline trains/evaluates on the
    same forecast setup. Windows with any missing step (input_mask==0) or a
    non-finite static feature are dropped.
    """
    T = X_dynamic.shape[0]
    node_count = X_dynamic.shape[1]
    static_dim = static.shape[1]
    feature_count = X_dynamic.shape[2] + static_dim

    empty = (
        np.empty((0, seq_len, feature_count), dtype=np.float32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
    )

    timestamps = np.asarray(timestamps, dtype=np.int64)
    min_target = seq_len - 1 + pred_horizon
    valid_t = timestamps[(timestamps >= min_target) & (timestamps <= T - 1)]
    if valid_t.size == 0:
        return empty

    static_ok = np.isfinite(static).all(axis=1)  # (N,)

    # Per (time, node) "fully observed" flag, then a sliding-window AND over the
    # seq_len input steps via a cumulative-sum trick (vectorized, no Python loop):
    # the window [le-seq_len+1 .. le] is fully observed iff its observed count
    # equals seq_len.
    obs = np.asarray(input_mask, dtype=bool).all(axis=2)  # (T, N)
    cum = np.concatenate(
        [np.zeros((1, node_count), dtype=np.int32), np.cumsum(obs, axis=0, dtype=np.int32)],
        axis=0,
    )  # (T+1, N)
    last_input_t = valid_t - pred_horizon  # (V,)
    win_obs = (cum[last_input_t + 1] - cum[last_input_t - seq_len + 1]) == seq_len  # (V, N)

    sel_rows = np.asarray(select[valid_t]).astype(bool)  # (V, N)
    keep = sel_rows & static_ok[None, :] & win_obs  # (V, N)
    if not keep.any():
        return empty

    # Row-major nonzero iterates (timestamp-order, node-order), matching the
    # original double-loop ordering so results are identical.
    row_idx, n_sel = np.nonzero(keep)
    t_sel = valid_t[row_idx]  # (R,)
    start_sel = t_sel - pred_horizon - seq_len + 1  # (R,)

    offsets = np.arange(seq_len, dtype=np.int64)
    time_idx = start_sel[:, None] + offsets[None, :]  # (R, seq_len)
    X_win = X_dynamic[time_idx, n_sel[:, None], :]  # (R, seq_len, F)
    static_win = np.broadcast_to(
        static[n_sel][:, None, :], (t_sel.shape[0], seq_len, static_dim)
    )  # (R, seq_len, static)
    X_out = np.concatenate([X_win, static_win], axis=2).astype(np.float32)

    return (
        X_out,
        t_sel.astype(np.int32),
        n_sel.astype(np.int32),
    )


def build_sequence_rows(
    X_dynamic: np.ndarray,
    input_mask: np.ndarray,
    z: np.ndarray,
    z_mask: np.ndarray,
    static: np.ndarray,
    timestamps: np.ndarray,
    seq_len: int = 12,
    pred_horizon: int = 1,
) -> GRUSplit:
    X_out, time_flat, node_flat = _build_sequence_windows(
        X_dynamic, input_mask, z_mask, static, timestamps, seq_len, pred_horizon
    )
    if len(time_flat):
        y_out = np.asarray(z[time_flat, node_flat], dtype=np.uint8)
    else:
        y_out = np.array([], dtype=np.uint8)
    return GRUSplit(X=X_out, y=y_out, global_t=time_flat, node_idx=node_flat)



def build_y_state_rows(
    X_dynamic: np.ndarray,
    input_mask: np.ndarray,
    y_state: np.ndarray,
    static: np.ndarray,
    timestamps: np.ndarray,
    seq_len: int = 12,
    pred_horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagnostic windows for cells already in the paralysis state (y==1).

    Returns 3D sequences (not flat rows) so the GRU can score them directly,
    using the same window policy as ``build_sequence_rows``.
    """
    return _build_sequence_windows(
        X_dynamic, input_mask, y_state, static, timestamps, seq_len, pred_horizon
    )


def load_fold_data(
    data_dir: Path,
    percentile: str,
    fold: str,
    positive_timestamp_ratio: float,
    seed: int,
    seq_len: int = 12,
    pred_horizon: int = 1,
) -> FoldData:

    X_dynamic = np.asarray(np.load(data_dir / "features/X_dynamic.npy"), dtype=np.float32)
    input_mask = np.load(data_dir / "features/input_mask.npy", mmap_mode="r")
    continuous_prev_hour = np.load(data_dir / "features/continuous_prev_hour.npy", mmap_mode="r").astype(bool)
    A = np.load(data_dir / "graph/A.npy", mmap_mode="r")

    adjacency = build_adjacency_list(np.asarray(A))
    node_ids = load_node_ids(data_dir / "graph/node_ids.csv")

    target_dir = data_dir / "labels" / percentile / fold
    y_state = np.load(target_dir / "y.npy", mmap_mode="r")
    z = np.load(target_dir / "z.npy", mmap_mode="r")
    z_mask = np.load(target_dir / "z_mask.npy", mmap_mode="r")
    split_code = np.asarray(np.load(target_dir / "split_code.npy", mmap_mode="r"))
    static, static_features = load_static(target_dir / "X_static.csv", node_ids)

    # Z-score dynamic features using train-split timestamps only (no leakage),
    # mirroring the STGCN loader so GRU sees comparably scaled inputs.
    train_mask = split_code == SPLIT_TO_CODE["train"]
    X_train_view = X_dynamic[train_mask]
    mean = np.nanmean(X_train_view, axis=(0, 1), keepdims=True)
    std = np.nanstd(X_train_view, axis=(0, 1), keepdims=True)
    std_safe = np.where(std > 1e-6, std, 1.0)
    X_norm = (X_dynamic - mean) / std_safe
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    selected_train_ts, train_summary = sample_train_timestamps(
        split_code=split_code,
        z=np.asarray(z),
        z_mask=np.asarray(z_mask),
        positive_timestamp_ratio=positive_timestamp_ratio,
        seed=seed,
    )
    val_ts = np.flatnonzero(split_code == SPLIT_TO_CODE["val"]).astype(np.int32)
    test_ts = np.flatnonzero(split_code == SPLIT_TO_CODE["test"]).astype(np.int32)

    train_split = build_sequence_rows(X_norm, input_mask, z, z_mask, static, selected_train_ts, seq_len, pred_horizon)
    val_split = build_sequence_rows(X_norm, input_mask, z, z_mask, static, val_ts, seq_len, pred_horizon)
    test_split = build_sequence_rows(X_norm, input_mask, z, z_mask, static, test_ts, seq_len, pred_horizon)
    y_state_val_X, _yval_t, _yval_n = build_y_state_rows(X_norm, input_mask, y_state, static, val_ts, seq_len, pred_horizon)
    y_state_test_X, _ytest_t, _ytest_n = build_y_state_rows(X_norm, input_mask, y_state, static, test_ts, seq_len, pred_horizon)

    dataset_summary = {
        "percentile": percentile,
        "fold": fold,
        "seq_len": int(seq_len),
        "pred_horizon": int(pred_horizon),
        "features": {
            "dynamic_count": int(X_dynamic.shape[2]),
            "static_features": static_features,
            "total_sequence_features": (
                int(train_split.X.shape[2])
                if train_split.X.ndim == 3
                else 0
            ),
        },
        "normalization": {
            "method": "z-score",
            "computed_on": "train-split timestamps only",
            "mean": mean.reshape(-1).tolist(),
            "std": std.reshape(-1).tolist(),
        },
        "sampling": train_summary,
        "rows": {
            "train": {"n": int(len(train_split.y)), "positive": int(train_split.y.sum())},
            "val": {"n": int(len(val_split.y)), "positive": int(val_split.y.sum())},
            "test": {"n": int(len(test_split.y)), "positive": int(test_split.y.sum())},
            "y1_val": int(len(y_state_val_X)),
            "y1_test": int(len(y_state_test_X)),
        },
        "missing_policy": "drop rows unless z_mask==1 and all dynamic/static input features are finite over the window",
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

