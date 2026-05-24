#!/usr/bin/env python3
"""Run tabular baselines on train-ready D7 data.

Model-specific code lives in `flood_traffic.baselines.*`; this file only
loads fold data, dispatches models, and writes experiment outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from flood_traffic.baselines import logistic_regression, xgboost_model
from flood_traffic.constants import FOLDS, PERCENTILES
from flood_traffic.evaluation import evaluate_model_outputs
from flood_traffic.io_utils import append_csv, write_json
from flood_traffic.reporting import result_row
from flood_traffic.tabular_data import FoldData, load_fold_data


def fit_baseline(
    model_key: str,
    fold_data: FoldData,
    args: argparse.Namespace,
) -> tuple[str, Any, dict[str, Any], Any]:
    if model_key == "logistic":
        model, model_info = logistic_regression.fit(
            train_split=fold_data.train,
            val_split=fold_data.val,
            C_values=args.logistic_C,
            max_iter=args.logistic_max_iter,
            seed=args.seed,
        )
        return logistic_regression.MODEL_NAME, model, model_info, logistic_regression.predict

    if model_key == "xgboost":
        model, model_info = xgboost_model.fit(
            train_split=fold_data.train,
            val_split=fold_data.val,
            seed=args.seed,
            num_boost_round=args.xgb_rounds,
            early_stopping_rounds=args.xgb_early_stopping_rounds,
        )
        return xgboost_model.MODEL_NAME, model, model_info, xgboost_model.predict

    raise ValueError(f"Unknown model key: {model_key}")


def maybe_save_model(model_key: str, model: Any, run_dir: Path) -> None:
    if model_key == "xgboost":
        xgboost_model.save_model(model, run_dir / "xgboost_model.json")


def run_one(
    data_dir: Path,
    out_dir: Path,
    percentile: str,
    fold: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    print(f"[run] {percentile} {fold}", flush=True)
    fold_data = load_fold_data(
        data_dir=data_dir,
        percentile=percentile,
        fold=fold,
        positive_timestamp_ratio=args.positive_timestamp_ratio,
        seed=args.seed,
    )

    run_dir = out_dir / percentile / fold
    write_json(run_dir / "dataset_summary.json", fold_data.dataset_summary)

    rows: list[dict[str, Any]] = []
    for model_key in args.models:
        print(f"[{model_key}] fitting {percentile} {fold}", flush=True)
        model_name, model, model_info, predict_fn = fit_baseline(model_key, fold_data, args)
        val_metrics, test_metrics, tau = evaluate_model_outputs(
            model=model,
            predict_fn=predict_fn,
            val_split=fold_data.val,
            test_split=fold_data.test,
            y_state_val_X=fold_data.y_state_val_X,
            y_state_test_X=fold_data.y_state_test_X,
            min_precision=args.min_precision,
            adjacency=fold_data.adjacency,
            continuous_prev_hour=fold_data.continuous_prev_hour,
        )
        model_info["tau"] = tau
        write_json(run_dir / f"{model_name}_metrics.json", {"val": val_metrics, "test": test_metrics, "model": model_info})
        maybe_save_model(model_key, model, run_dir)
        rows.append(result_row(model_name, percentile, fold, "val", val_metrics, fold_data.train_summary, model_info))
        rows.append(result_row(model_name, percentile, fold, "test", test_metrics, fold_data.train_summary, model_info))

    return rows


def parse_args() -> argparse.Namespace:
    data_root = Path(__file__).resolve().parents[1] / "data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=data_root / "train_ready/d7_active_giant_full")
    parser.add_argument("--out-dir", type=Path, default=data_root / "outputs/tabular_baselines")
    parser.add_argument("--percentiles", nargs="+", default=PERCENTILES)
    parser.add_argument("--folds", nargs="+", default=FOLDS)
    parser.add_argument("--models", nargs="+", default=["logistic", "xgboost"], choices=["logistic", "xgboost"])
    parser.add_argument("--positive-timestamp-ratio", type=float, default=0.20)
    parser.add_argument("--min-precision", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logistic-C", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--logistic-max-iter", type=int, default=1000)
    parser.add_argument("--xgb-rounds", type=int, default=1000)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for percentile in args.percentiles:
        for fold in args.folds:
            rows = run_one(args.data_dir, args.out_dir, percentile, fold, args)
            all_rows.extend(rows)
            append_csv(args.out_dir / "metrics_summary.csv", rows)
    write_json(args.out_dir / "run_config.json", vars(args))
    print(f"[done] wrote {args.out_dir / 'metrics_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
