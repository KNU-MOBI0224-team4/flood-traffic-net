#!/usr/bin/env python3
"""Run STGCN training on train-ready D7 data.

Mirrors run_tabular_baselines.py: loads each (percentile, fold) once,
fits an STGCN, evaluates with the same metrics as the tabular baselines,
and writes per-fold JSON + a cumulative metrics_summary.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from flood_traffic.constants import FOLDS, PERCENTILES
from flood_traffic.graph_data import STGCNFoldData, load_graph_fold_data
from flood_traffic.io_utils import append_csv, write_json
from flood_traffic.reporting import result_row
from flood_traffic.stgcn import trainer
from flood_traffic.stgcn.evaluator import evaluate_stgcn_outputs


def fit_stgcn(fold_graph: STGCNFoldData, args: argparse.Namespace) -> tuple[str, Any, dict[str, Any]]:
    in_channels = int(fold_graph.train_dataset.X.shape[-1])
    num_nodes = int(fold_graph.A_hat.shape[0])
    model, info = trainer.fit(
        train_dataset=fold_graph.train_dataset,
        val_dataset=fold_graph.val_dataset,
        A_hat=fold_graph.A_hat,
        static_features=fold_graph.static,
        num_nodes=num_nodes,
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        static_embedding_dim=args.static_embedding_dim,
        cheb_k=args.cheb_k,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        pos_weight_cap=args.pos_weight_cap,
        grad_clip=args.grad_clip,
        device=args.device,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        loss_type=args.loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    return trainer.MODEL_NAME, model, info


def run_one(
    data_dir: Path,
    out_dir: Path,
    percentile: str,
    fold: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    print(f"[run] {percentile} {fold}", flush=True)
    fold_graph = load_graph_fold_data(
        data_dir=data_dir,
        percentile=percentile,
        fold=fold,
        seq_len=args.seq_len,
        pred_horizon=args.pred_horizon,
        positive_timestamp_ratio=args.positive_timestamp_ratio,
        seed=args.seed,
        use_static=args.static,
        use_input_mask=args.input_mask,
        use_time_features=args.time_features,
    )

    run_dir = out_dir / percentile / fold
    write_json(run_dir / "dataset_summary.json", fold_graph.dataset_summary)

    print(f"[stgcn] fitting {percentile} {fold}", flush=True)
    model_name, model, model_info = fit_stgcn(fold_graph, args)

    val_metrics, test_metrics, tau = evaluate_stgcn_outputs(
        model=model,
        A_hat=fold_graph.A_hat,
        static_features=fold_graph.static,
        val_dataset=fold_graph.val_dataset,
        test_dataset=fold_graph.test_dataset,
        val_split=fold_graph.val_split,
        test_split=fold_graph.test_split,
        val_y_state_t=fold_graph.val_y_state_t,
        val_y_state_n=fold_graph.val_y_state_n,
        test_y_state_t=fold_graph.test_y_state_t,
        test_y_state_n=fold_graph.test_y_state_n,
        min_precision=args.min_precision,
        adjacency=fold_graph.adjacency,
        continuous_prev_hour=fold_graph.continuous_prev_hour,
        batch_size=args.batch_size,
        device=args.device,
    )
    model_info["tau"] = tau
    write_json(
        run_dir / f"{model_name}_metrics.json",
        {"val": val_metrics, "test": test_metrics, "model": model_info},
    )
    trainer.save_model(model, run_dir / f"{model_name}_model.pt")

    return [
        result_row(model_name, percentile, fold, "val", val_metrics, fold_graph.train_summary, model_info),
        result_row(model_name, percentile, fold, "test", test_metrics, fold_graph.train_summary, model_info),
    ]


def parse_args() -> argparse.Namespace:
    data_root = Path(__file__).resolve().parents[1] / "data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=data_root / "train_ready/d7_active_giant_full")
    parser.add_argument("--out-dir", type=Path, default=data_root / "outputs/stgcn")
    parser.add_argument("--percentiles", nargs="+", default=PERCENTILES)
    parser.add_argument("--folds", nargs="+", default=FOLDS)
    parser.add_argument("--positive-timestamp-ratio", type=float, default=0.20)
    parser.add_argument("--min-precision", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--pred-horizon", type=int, default=1)
    parser.add_argument(
        "--static",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject static node features at the GCN input initialization stage.",
    )
    parser.add_argument(
        "--input-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include input_mask as extra channels so the model can tell imputed values from observed ones.",
    )
    parser.add_argument(
        "--time-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include cyclical time features (sin/cos of hour, dow, month) as extra channels.",
    )
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--out-channels", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--static-embedding-dim",
        type=int,
        default=8,
        help="Embedding dimension for static node features before GCN propagation.",
    )
    parser.add_argument(
        "--cheb-k",
        type=int,
        default=3,
        help="Number of Chebyshev polynomial supports. The spatial layer always uses Chebyshev graph convolution.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pos-weight-cap", type=float, default=50.0)
    parser.add_argument(
        "--loss",
        type=str,
        choices=["bce", "focal"],
        default="bce",
        help="Loss function. 'bce' = BCE with pos_weight; 'focal' = Focal Loss (ignores pos_weight).",
    )
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop if val_auprc does not improve for N epochs. 0 disables.",
    )
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
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
