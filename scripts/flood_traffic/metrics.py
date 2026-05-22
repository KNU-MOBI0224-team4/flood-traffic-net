"""Shared metrics for onset prediction models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def safe_metric(fn: Any, y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2 and fn is roc_auc_score:
            return float("nan")
        return float(fn(y_true, score))
    except Exception:
        return float("nan")


def select_tau_at_precision(
    y_true: np.ndarray,
    score: np.ndarray,
    min_precision: float = 0.10,
) -> dict[str, Any]:
    y = y_true.astype(np.uint8)
    s = score.astype(np.float64)
    if y.size == 0 or int(y.sum()) == 0:
        return {
            "tau": float("inf"),
            "precision": 0.0,
            "recall": 0.0,
            "k": 0,
            "feasible": False,
        }

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tp = np.cumsum(y_sorted == 1)
    rank = np.arange(1, len(y_sorted) + 1)
    precision = tp / rank
    recall = tp / max(1, int(y.sum()))
    feasible_idx = np.flatnonzero(precision >= min_precision)
    if feasible_idx.size == 0:
        best = 0
        return {
            "tau": float(s_sorted[best]),
            "precision": float(precision[best]),
            "recall": float(recall[best]),
            "k": int(best + 1),
            "feasible": False,
        }
    best_local = feasible_idx[np.argmax(recall[feasible_idx])]
    return {
        "tau": float(s_sorted[best_local]),
        "precision": float(precision[best_local]),
        "recall": float(recall[best_local]),
        "k": int(best_local + 1),
        "feasible": True,
    }


def threshold_metrics(y_true: np.ndarray, score: np.ndarray, tau: float) -> dict[str, Any]:
    pred = score >= tau
    y = y_true.astype(bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision_tau": precision,
        "recall_tau": recall,
        "f1_tau": f1,
        "tp_tau": tp,
        "fp_tau": fp,
        "fn_tau": fn,
        "tn_tau": tn,
        "pred_pos_rate_tau": float(pred.mean()) if len(pred) else 0.0,
    }


def evaluate_binary(
    y_true: np.ndarray,
    score: np.ndarray,
    tau: float | None,
    min_precision: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = {
        "n": int(len(y_true)),
        "positive": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "auprc": safe_metric(average_precision_score, y_true, score),
        "roc_auc": safe_metric(roc_auc_score, y_true, score),
        "brier": safe_metric(brier_score_loss, y_true, score),
    }
    tau_info = (
        select_tau_at_precision(y_true, score, min_precision=min_precision)
        if tau is None
        else {
            "tau": tau,
            "precision": float("nan"),
            "recall": float("nan"),
            "k": -1,
            "feasible": True,
        }
    )
    metrics.update(threshold_metrics(y_true, score, float(tau_info["tau"])))
    return metrics, tau_info


def build_adjacency_list(A: np.ndarray) -> list[list[int]]:
    return [np.flatnonzero(A[idx]).astype(int).tolist() for idx in range(A.shape[0])]


def strict_event_recall(
    y_true: np.ndarray,
    score: np.ndarray,
    global_t: np.ndarray,
    node_idx: np.ndarray,
    tau: float,
    adjacency: list[list[int]],
    continuous_prev_hour: np.ndarray,
) -> dict[str, Any]:
    positive_positions = np.flatnonzero(y_true == 1)
    if positive_positions.size == 0:
        return {
            "event_recall_tau": float("nan"),
            "event_count": 0,
            "event_hit_count": 0,
            "positive_timestamp_hit_rate_tau": float("nan"),
            "positive_timestamp_count": 0,
            "positive_timestamp_hit_count": 0,
        }

    pos_cells = set()
    score_by_cell: dict[tuple[int, int], float] = {}
    for pos in positive_positions:
        cell = (int(global_t[pos]), int(node_idx[pos]))
        pos_cells.add(cell)
        score_by_cell[cell] = float(score[pos])

    visited: set[tuple[int, int]] = set()
    event_count = 0
    event_hit_count = 0
    for start in pos_cells:
        if start in visited:
            continue
        event_count += 1
        stack = [start]
        visited.add(start)
        hit = False
        while stack:
            t_idx, n_idx = stack.pop()
            if score_by_cell[(t_idx, n_idx)] >= tau:
                hit = True
            for neigh in adjacency[n_idx]:
                nxt = (t_idx, int(neigh))
                if nxt in pos_cells and nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
            prev_cell = (t_idx - 1, n_idx)
            if t_idx > 0 and bool(continuous_prev_hour[t_idx]) and prev_cell in pos_cells and prev_cell not in visited:
                visited.add(prev_cell)
                stack.append(prev_cell)
            next_cell = (t_idx + 1, n_idx)
            if (
                t_idx + 1 < len(continuous_prev_hour)
                and bool(continuous_prev_hour[t_idx + 1])
                and next_cell in pos_cells
                and next_cell not in visited
            ):
                visited.add(next_cell)
                stack.append(next_cell)
        event_hit_count += int(hit)

    by_timestamp: dict[int, list[int]] = {}
    for pos in positive_positions:
        by_timestamp.setdefault(int(global_t[pos]), []).append(int(pos))
    timestamp_hit = 0
    for indices in by_timestamp.values():
        timestamp_hit += int(np.any(score[indices] >= tau))

    return {
        "event_recall_tau": event_hit_count / event_count if event_count else float("nan"),
        "event_count": event_count,
        "event_hit_count": event_hit_count,
        "positive_timestamp_hit_rate_tau": timestamp_hit / len(by_timestamp) if by_timestamp else float("nan"),
        "positive_timestamp_count": len(by_timestamp),
        "positive_timestamp_hit_count": timestamp_hit,
    }

