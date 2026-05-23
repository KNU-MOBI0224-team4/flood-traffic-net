#!/usr/bin/env python3 
"""Run GRU baseline on train-ready D7 data.""" 
from __future__ import annotations 
import argparse 
import sys 
from pathlib import Path 
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent)) 
from flood_traffic.baselines import gru 
from flood_traffic.constants import FOLDS, PERCENTILES 
from flood_traffic.evaluation import evaluate_model_outputs 
from flood_traffic.io_utils import append_csv, write_json 
from flood_traffic.reporting import result_row 
from flood_traffic.sequence_data import load_fold_data


def fit_baseline( 
        fold_data, 
        args: argparse.Namespace, 
) -> tuple[str, Any, dict[str, Any], Any]: 
    model, model_info = gru.fit( 
        train_split=fold_data.train, 
        val_split=fold_data.val, 
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers, 
        dropout=args.dropout, 
        learning_rate=args.learning_rate, 
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        seed=args.seed, 
        device=args.device, ) 
    return ( gru.MODEL_NAME, model, model_info, gru.predict, )

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

    write_json( run_dir / "dataset_summary.json", fold_data.dataset_summary, )
    print(f"[gru] fitting {percentile} {fold}", flush=True)
    model_name, model, model_info, predict_fn = fit_baseline( fold_data, args, )
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
    write_json(
        run_dir / f"{model_name}_metrics.json",
        {
            "val": val_metrics,
            "test": test_metrics,
            "model": model_info,
        },
    )
    rows = []
    rows.append(
        result_row(
            model_name,
            percentile,
            fold,
            "val",
            val_metrics,
            fold_data.train_summary,
            model_info,
        )
    )
    rows.append(
        result_row(
            model_name,
            percentile, fold,
            "test", test_metrics,
            fold_data.train_summary,
            model_info,
        )
    )
    return rows

def parse_args() -> argparse.Namespace:
    data_root = Path(__file__).resolve().parents[1] / "data"
    parser = argparse.ArgumentParser()
    parser.add_argument( "--data-dir", type=Path, default=data_root / "data_train/d7_active_giant_2016_01", )
    parser.add_argument( "--out-dir", type=Path, default=data_root / "outputs/gru_baseline", )
    parser.add_argument( "--percentiles", nargs="+", default=["p99"], )
    parser.add_argument( "--folds", nargs="+", default=["fold_1_train2016_2020_val2021_test2022"], )
    parser.add_argument( "--positive-timestamp-ratio", type=float, default=0.20, )
    parser.add_argument( "--min-precision", type=float, default=0.10, )
    parser.add_argument( "--seed", type=int, default=42, )
    parser.add_argument( "--hidden-size", type=int, default=32, )
    parser.add_argument( "--num-layers", type=int, default=1, )
    parser.add_argument( "--dropout", type=float, default=0.0, )
    parser.add_argument( "--learning-rate", type=float, default=1e-3, )
    parser.add_argument( "--batch-size", type=int, default=256, )
    parser.add_argument( "--epochs", type=int, default=5, )
    parser.add_argument( "--device", type=str, default="cpu", )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for percentile in args.percentiles:
        for fold in args.folds:

            rows = run_one(
                args.data_dir,
                args.out_dir,
                percentile,
                fold,
                args,
            )
            all_rows.extend(rows)
            append_csv( args.out_dir / "metrics_summary.csv", rows, )
    write_json( args.out_dir / "run_config.json", vars(args), )
    print( f"[done] wrote {args.out_dir / 'metrics_summary.csv'}", flush=True, )


if __name__ == "__main__": main()