"""Shared timestamp sampling policies for rare-onset training."""

from __future__ import annotations

from typing import Any

import numpy as np

from flood_traffic.constants import SPLIT_TO_CODE


def sample_train_timestamps(
    split_code: np.ndarray,
    z: np.ndarray,
    z_mask: np.ndarray,
    positive_timestamp_ratio: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select train timestamps while preserving all positive-onset timestamps.

    STGCN/GRU-style graph models train by timestamp, not by independent
    node-row sampling. This policy is therefore timestamp-level: include every
    train timestamp containing at least one usable z=1 node, then sample
    negative-only timestamps to reach the requested positive timestamp ratio.
    """

    if not (0 < positive_timestamp_ratio < 1):
        raise ValueError("positive_timestamp_ratio must be in (0, 1)")

    train_ts = np.flatnonzero(split_code == SPLIT_TO_CODE["train"])
    pos_any = ((z[train_ts] == 1) & (z_mask[train_ts] == 1)).any(axis=1)
    positive_ts = train_ts[pos_any]
    negative_ts = train_ts[~pos_any]
    rng = np.random.default_rng(seed)

    if len(positive_ts) == 0:
        selected_negative = negative_ts
    else:
        n_negative = int(round(len(positive_ts) * (1.0 - positive_timestamp_ratio) / positive_timestamp_ratio))
        n_negative = min(n_negative, len(negative_ts))
        selected_negative = rng.choice(negative_ts, size=n_negative, replace=False)
    selected = np.concatenate([positive_ts, selected_negative])
    selected.sort()
    summary = {
        "train_timestamps_total": int(len(train_ts)),
        "positive_timestamps_total": int(len(positive_ts)),
        "negative_timestamps_total": int(len(negative_ts)),
        "selected_timestamps": int(len(selected)),
        "selected_positive_timestamps": int(len(positive_ts)),
        "selected_negative_timestamps": int(len(selected_negative)),
        "selected_positive_timestamp_ratio": float(len(positive_ts) / len(selected)) if len(selected) else 0.0,
        "sampling_policy": "all positive timestamps + sampled negative timestamps to reach requested positive timestamp ratio",
        "requested_positive_timestamp_ratio": float(positive_timestamp_ratio),
    }
    return selected.astype(np.int32), summary

