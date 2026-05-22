"""XGBoost baseline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from flood_traffic.tabular_data import TabularSplit

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - handled at runtime
    xgb = None


MODEL_NAME = "xgboost"


def fit(
    train_split: TabularSplit,
    val_split: TabularSplit,
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[Any, dict[str, Any]]:
    if xgb is None:
        raise RuntimeError("xgboost is not installed")

    dtrain = xgb.DMatrix(train_split.X, label=train_split.y)
    dval = xgb.DMatrix(val_split.X, label=val_split.y)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "lambda": 1.0,
        "alpha": 0.0,
        "scale_pos_weight": 1.0,
        "seed": seed,
        "nthread": 0,
    }
    evals_result: dict[str, Any] = {}
    start = time.time()
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=25,
        evals_result=evals_result,
    )
    info = {
        "params": params,
        "best_iteration": int(model.best_iteration),
        "best_score": float(model.best_score),
        "elapsed_sec": time.time() - start,
        "num_boost_round": int(num_boost_round),
        "early_stopping_rounds": int(early_stopping_rounds),
        "evals_result": evals_result,
    }
    return model, info


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    if xgb is None:
        raise RuntimeError("xgboost is not installed")
    dmat = xgb.DMatrix(X)
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return model.predict(dmat).astype(np.float64)
    return model.predict(dmat, iteration_range=(0, int(best_iteration) + 1)).astype(np.float64)


def save_model(model: Any, path: Path) -> None:
    model.save_model(str(path))

