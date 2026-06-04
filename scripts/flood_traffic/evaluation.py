"""Shared evaluation workflow for baseline and future models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from flood_traffic.io_utils import save_test_predictions
from flood_traffic.metrics import evaluate_binary, strict_event_recall
from flood_traffic.tabular_data import TabularSplit


PredictFn = Callable[[Any, np.ndarray], np.ndarray]


def evaluate_model_outputs(
    model: Any,
    predict_fn: PredictFn,
    val_split: TabularSplit,
    test_split: TabularSplit,
    y_state_val_X: np.ndarray,
    y_state_test_X: np.ndarray,
    min_precision: float,
    adjacency: list[list[int]],
    continuous_prev_hour: np.ndarray,
    test_pred_out: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    val_score = predict_fn(model, val_split.X)
    val_metrics, tau_info = evaluate_binary(val_split.y, val_score, tau=None, min_precision=min_precision)
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

    test_score = predict_fn(model, test_split.X)
    if test_pred_out is not None:
        save_test_predictions(
            test_pred_out, test_split.global_t, test_split.node_idx, test_split.y, test_score
        )
    test_metrics, _tau = evaluate_binary(test_split.y, test_score, tau=tau, min_precision=min_precision)
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

    for prefix, X_diag, metrics in [
        ("y1_val", y_state_val_X, val_metrics),
        ("y1_test", y_state_test_X, test_metrics),
    ]:
        if len(X_diag) == 0:
            metrics[f"{prefix}_n"] = 0
            metrics[f"{prefix}_score_mean"] = float("nan")
            metrics[f"{prefix}_score_p95"] = float("nan")
            metrics[f"{prefix}_alarm_rate_tau"] = float("nan")
            continue
        diag_score = predict_fn(model, X_diag)
        metrics[f"{prefix}_n"] = int(len(diag_score))
        metrics[f"{prefix}_score_mean"] = float(np.mean(diag_score))
        metrics[f"{prefix}_score_p95"] = float(np.quantile(diag_score, 0.95))
        metrics[f"{prefix}_alarm_rate_tau"] = float(np.mean(diag_score >= tau))

    return val_metrics, test_metrics, tau

