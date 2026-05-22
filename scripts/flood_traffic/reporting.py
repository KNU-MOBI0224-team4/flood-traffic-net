"""Output row formatting for experiment summaries."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


METRIC_ROW_KEYS = [
    "n",
    "positive",
    "positive_rate",
    "auprc",
    "roc_auc",
    "brier",
    "precision_tau",
    "recall_tau",
    "f1_tau",
    "tp_tau",
    "fp_tau",
    "fn_tau",
    "tn_tau",
    "pred_pos_rate_tau",
    "event_recall_tau",
    "event_count",
    "event_hit_count",
    "positive_timestamp_hit_rate_tau",
    "positive_timestamp_count",
    "positive_timestamp_hit_count",
    "y1_val_n",
    "y1_val_score_mean",
    "y1_val_score_p95",
    "y1_val_alarm_rate_tau",
    "y1_test_n",
    "y1_test_score_mean",
    "y1_test_score_p95",
    "y1_test_alarm_rate_tau",
    "tau_select_tau",
    "tau_select_precision",
    "tau_select_recall",
    "tau_select_k",
    "tau_select_feasible",
]


def result_row(
    model_name: str,
    percentile: str,
    fold: str,
    split: str,
    metrics: dict[str, Any],
    train_summary: dict[str, Any],
    model_info: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "model": model_name,
        "percentile": percentile,
        "fold": fold,
        "split": split,
        "selected_train_timestamps": train_summary["selected_timestamps"],
        "selected_positive_timestamp_ratio": train_summary["selected_positive_timestamp_ratio"],
        "model_selection": json.dumps(
            {k: v for k, v in model_info.items() if k not in {"evals_result", "trials"}},
            default=str,
        ),
    }
    for key in METRIC_ROW_KEYS:
        value = metrics.get(key, "")
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        row[key] = value
    return row

