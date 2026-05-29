"""STGCN-specific evaluation orchestration.

Mirrors flood_traffic.evaluation.evaluate_model_outputs but adapted to STGCN
inference (sequence dataset + A_hat) while reusing the same metric primitives
in flood_traffic.metrics for fair comparison with the tabular baselines.

The caller must build val_split/test_split (and the y_state coords) using the
same target timestamps the STGCN dataset uses, so every queried (t, n) has a
corresponding prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from flood_traffic.graph_data import STGCNDataset
from flood_traffic.metrics import evaluate_binary, strict_event_recall
from flood_traffic.stgcn.stgcn import STGCN
from flood_traffic.stgcn.trainer import predict_score_matrix
from flood_traffic.tabular_data import TabularSplit


def _align_scores(
    target_ts: np.ndarray,
    score_matrix: np.ndarray,
    query_t: np.ndarray,
    query_n: np.ndarray,
) -> np.ndarray:
    """Return scores[t, n] for each (query_t[i], query_n[i]) via O(1) lookup."""
    if query_t.size == 0:
        return np.empty((0,), dtype=np.float64)
    lookup = np.full(int(target_ts.max()) + 1, -1, dtype=np.int64)
    lookup[target_ts.astype(int)] = np.arange(len(target_ts))
    rows = lookup[query_t.astype(int)]
    if not (rows >= 0).all():
        missing = int((rows < 0).sum())
        raise ValueError(
            f"{missing} query timestamps not present in dataset target_timestamps"
        )
    return score_matrix[rows, query_n.astype(int)].astype(np.float64)


def evaluate_stgcn_outputs(
    model: STGCN,
    A_hat: np.ndarray,
    static_features: np.ndarray | None,
    val_dataset: STGCNDataset,
    test_dataset: STGCNDataset,
    val_split: TabularSplit,
    test_split: TabularSplit,
    val_y_state_t: np.ndarray,
    val_y_state_n: np.ndarray,
    test_y_state_t: np.ndarray,
    test_y_state_n: np.ndarray,
    min_precision: float,
    adjacency: list[list[int]],
    continuous_prev_hour: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    val_target_ts, val_scores_mat = predict_score_matrix(
        model, val_dataset, A_hat, static_features, batch_size, device
    )
    test_target_ts, test_scores_mat = predict_score_matrix(
        model, test_dataset, A_hat, static_features, batch_size, device
    )

    val_score = _align_scores(
        val_target_ts, val_scores_mat, val_split.global_t, val_split.node_idx
    )
    test_score = _align_scores(
        test_target_ts, test_scores_mat, test_split.global_t, test_split.node_idx
    )

    val_metrics, tau_info = evaluate_binary(
        val_split.y, val_score, tau=None, min_precision=min_precision
    )
    tau = float(tau_info["tau"])
    val_metrics.update({f"tau_select_{k}": v for k, v in tau_info.items()})
    val_metrics.update(
        strict_event_recall(
            val_split.y,
            val_score,
            val_split.global_t,
            val_split.node_idx,
            tau,
            adjacency,
            continuous_prev_hour,
        )
    )

    test_metrics, _ = evaluate_binary(
        test_split.y, test_score, tau=tau, min_precision=min_precision
    )
    test_metrics.update(
        strict_event_recall(
            test_split.y,
            test_score,
            test_split.global_t,
            test_split.node_idx,
            tau,
            adjacency,
            continuous_prev_hour,
        )
    )

    diag_inputs = [
        ("y1_val", val_y_state_t, val_y_state_n, val_target_ts, val_scores_mat, val_metrics),
        ("y1_test", test_y_state_t, test_y_state_n, test_target_ts, test_scores_mat, test_metrics),
    ]
    for prefix, t_arr, n_arr, target_ts, scores_mat, metrics in diag_inputs:
        if len(t_arr) == 0:
            metrics[f"{prefix}_n"] = 0
            metrics[f"{prefix}_score_mean"] = float("nan")
            metrics[f"{prefix}_score_p95"] = float("nan")
            metrics[f"{prefix}_alarm_rate_tau"] = float("nan")
            continue
        diag_score = _align_scores(target_ts, scores_mat, t_arr, n_arr)
        metrics[f"{prefix}_n"] = int(len(diag_score))
        metrics[f"{prefix}_score_mean"] = float(np.mean(diag_score))
        metrics[f"{prefix}_score_p95"] = float(np.quantile(diag_score, 0.95))
        metrics[f"{prefix}_alarm_rate_tau"] = float(np.mean(diag_score >= tau))

    return val_metrics, test_metrics, tau
