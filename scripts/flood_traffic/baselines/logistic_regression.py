"""Logistic Regression baseline."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from flood_traffic.metrics import safe_metric
from flood_traffic.tabular_data import TabularSplit


MODEL_NAME = "logistic_regression"


def fit(
    train_split: TabularSplit,
    val_split: TabularSplit,
    C_values: list[float],
    max_iter: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    best_model = None
    best_info: dict[str, Any] = {"best_val_auprc": -math.inf}
    trials: list[dict[str, Any]] = []
    for C in C_values:
        start = time.time()
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(C),
                penalty="l2",
                solver="liblinear",
                max_iter=max_iter,
                class_weight=None,
                random_state=seed,
            ),
        )
        model.fit(train_split.X, train_split.y)
        score = model.predict_proba(val_split.X)[:, 1]
        auprc = safe_metric(average_precision_score, val_split.y, score)
        trial = {
            "C": float(C),
            "val_auprc": auprc,
            "elapsed_sec": time.time() - start,
        }
        trials.append(trial)
        if auprc > best_info["best_val_auprc"]:
            best_model = model
            best_info = {
                "best_C": float(C),
                "best_val_auprc": auprc,
                "trials": trials,
            }
    if best_model is None:
        raise RuntimeError("Failed to fit logistic regression")
    best_info["trials"] = trials
    return best_model, best_info


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float64)

