#!/usr/bin/env python3
"""Aggregate run_all_experiments outputs into two time-feature summary files.

Reads each model run's ``metrics_summary.csv`` plus its ``run_config.json``
(for the cheb_k / LayerNorm / time-feature config columns) and writes a tidy
``summary_time_off.csv`` and ``summary_time_on.csv`` under the base directory.

Only STGCN varies by time feature, so logistic/xgboost/gru run once (under
``shared/``) and their rows are included in both summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CONFIG_COLS = ["model", "cheb_k", "layernorm", "time_feature", "percentile", "fold", "split"]
TIDY_METRICS = [
    "n",
    "positive",
    "positive_rate",
    "auprc",
    "roc_auc",
    "brier",
    "precision_tau",
    "recall_tau",
    "f1_tau",
    "event_recall_tau",
    "event_count",
    "positive_timestamp_hit_rate_tau",
]


def _fmt_bool(value: object) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    return "na"


def read_run(run_dir: Path) -> list[dict]:
    """Read one run's metrics rows, tagged with its config columns."""
    metrics_csv = run_dir / "metrics_summary.csv"
    if not metrics_csv.exists():
        return []

    cfg: dict = {}
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cheb_k = cfg.get("cheb_k", "na")
    layernorm = _fmt_bool(cfg.get("hidden_layernorm"))
    time_feature = _fmt_bool(cfg.get("time_features"))

    rows: list[dict] = []
    with metrics_csv.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out = {
                "model": r.get("model", ""),
                "cheb_k": cheb_k,
                "layernorm": layernorm,
                "time_feature": time_feature,
                "percentile": r.get("percentile", ""),
                "fold": r.get("fold", ""),
                "split": r.get("split", ""),
            }
            for metric in TIDY_METRICS:
                out[metric] = r.get(metric, "")
            rows.append(out)
    return rows


def collect_branch(base: Path, time_branch: str) -> list[dict]:
    rows: list[dict] = []
    for k in (2, 3):
        for ln in ("on", "off"):
            rows += read_run(base / f"time_{time_branch}" / f"stgcn_k{k}_ln{ln}")
    # Models without time-feature support run once and are shared by both branches.
    rows += read_run(base / "shared" / "gru")
    rows += read_run(base / "shared" / "tabular")
    return rows


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CONFIG_COLS + TIDY_METRICS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("data/outputs/run_all"))
    args = parser.parse_args()

    for branch in ("off", "on"):
        rows = collect_branch(args.base, branch)
        out = args.base / f"summary_time_{branch}.csv"
        write_summary(out, rows)
        print(f"[aggregate] {out}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
