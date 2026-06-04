#!/usr/bin/env python3
"""Export case-study visualization data from run_all prediction dumps.

For each fold (test split), joins every model's per-cell test predictions with
ground truth (z onset, y state, z_mask), node coordinates, and timestamps, then
writes web-consumable artifacts. Requires the run_all sweep to have been run with
``--dump-predictions`` (see run_all_experiments.sh DUMP_PREDICTIONS=1).

Outputs under <out>/:
  nodes.json                  # node metadata + coordinates (fold-independent)
  <fold>/cells_full.csv.gz    # every evaluable test cell (offline, full fidelity)
  <fold>/cells_focus.json     # cells where z==1 OR any model alarmed (TP/FP/FN)
  <fold>/timeline.json        # per-timestamp counts (onsets, per-model alarm/TP/FP/FN)
  <fold>/meta.json            # fold, percentile, per-model tau, highlight model, counts

Confusion (per model, on z): TP=z1&alarm, FP=z0&alarm, FN=z1&no-alarm,
TN=z0&no-alarm, nodata=model had no prediction for that cell.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from flood_traffic.constants import SPLIT_TO_CODE
from flood_traffic.io_utils import load_node_ids, load_timestamps, read_csv_rows

# key -> (run subdir under base, model file prefix). All STGCN here are time-OFF.
MODELS = [
    ("stgcn_k2_lnon", "time_off/stgcn_k2_lnon", "stgcn"),
    ("stgcn_k2_lnoff", "time_off/stgcn_k2_lnoff", "stgcn"),
    ("stgcn_k3_lnon", "time_off/stgcn_k3_lnon", "stgcn"),
    ("stgcn_k3_lnoff", "time_off/stgcn_k3_lnoff", "stgcn"),
    ("gru", "shared/gru", "gru"),
    ("xgboost", "shared/tabular", "xgboost"),
    ("logistic", "shared/tabular", "logistic_regression"),
]
HIGHLIGHT_MODEL = "stgcn_k3_lnon"


def load_node_meta(data_dir: Path, meta_nodes_csv: Path) -> list[dict]:
    """node_idx-ordered metadata, joining full node_ids with the geometry CSV."""
    node_ids = load_node_ids(data_dir / "graph/node_ids.csv")  # node_idx -> segment_id
    meta_by_seg = {r["segment_id"]: r for r in read_csv_rows(meta_nodes_csv)}
    nodes = []
    missing = 0
    for idx, seg in enumerate(node_ids):
        m = meta_by_seg.get(seg, {})
        if not m:
            missing += 1

        def num(key):
            try:
                return float(m[key])
            except Exception:
                return None

        nodes.append(
            {
                "node_idx": idx,
                "segment_id": seg,
                "lon": num("centroid_lon"),
                "lat": num("centroid_lat"),
                "route": m.get("route"),
                "length_m": num("length_m"),
                "direction": m.get("direction"),
            }
        )
    if missing:
        print(f"[warn] {missing} nodes had no geometry match", flush=True)
    return nodes


def load_tau(run_dir: Path, prefix: str) -> float | None:
    p = run_dir / f"{prefix}_metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get("model", {}).get("tau")


def load_dump(run_dir: Path, prefix: str) -> np.ndarray | None:
    """Return (rows, 4) array global_t,node_idx,y_true,score, or None if absent."""
    p = run_dir / f"{prefix}_test_predictions.csv.gz"
    if not p.exists():
        return None
    arr = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
    return arr if arr.size else None


def confusion(score: float | None, alarm: bool | None, z: int) -> str:
    if score is None:
        return "nodata"
    if z == 1:
        return "TP" if alarm else "FN"
    return "FP" if alarm else "TN"


def export_fold(base: Path, data_dir: Path, percentile: str, fold: str, out: Path, write_full: bool) -> dict:
    label_dir = data_dir / "labels" / percentile / fold
    z = np.asarray(np.load(label_dir / "z.npy", mmap_mode="r"))
    y = np.asarray(np.load(label_dir / "y.npy", mmap_mode="r"))
    z_mask = np.asarray(np.load(label_dir / "z_mask.npy", mmap_mode="r"))
    split_code = np.asarray(np.load(label_dir / "split_code.npy", mmap_mode="r"))

    test_t = np.flatnonzero(split_code == SPLIT_TO_CODE["test"]).astype(np.int64)
    if test_t.size == 0:
        raise ValueError(f"no test timestamps for {percentile}/{fold}")
    N = z.shape[1]
    z_t = z[test_t].astype(np.int8)
    y_t = y[test_t].astype(np.int8)
    zmask_t = z_mask[test_t].astype(np.int8)

    # Per-model score matrix over (test timestamp row, node), NaN where no prediction.
    scores: dict[str, np.ndarray] = {}
    taus: dict[str, float | None] = {}
    present: list[str] = []
    for key, subdir, prefix in MODELS:
        run_dir = base / subdir / percentile / fold
        dump = load_dump(run_dir, prefix)
        taus[key] = load_tau(run_dir, prefix)
        mat = np.full((test_t.size, N), np.nan, dtype=np.float32)
        if dump is not None:
            g = dump[:, 0].astype(np.int64)
            n = dump[:, 1].astype(np.int64)
            s = dump[:, 3].astype(np.float32)
            rows = np.searchsorted(test_t, g)
            ok = (rows < test_t.size) & (test_t[np.clip(rows, 0, test_t.size - 1)] == g)
            mat[rows[ok], n[ok]] = s[ok]
            present.append(key)
        else:
            print(f"[warn] missing dump: {run_dir}/{prefix}_test_predictions.csv.gz", flush=True)
        scores[key] = mat

    # Universe = any model predicted OR an actual onset (z==1) in test.
    coverage = np.zeros((test_t.size, N), dtype=bool)
    for key in present:
        coverage |= ~np.isnan(scores[key])
    universe = coverage | (z_t == 1)
    ri, ci = np.nonzero(universe)
    print(f"[{fold}] test cells in universe: {ri.size}", flush=True)

    timestamps = load_timestamps(data_dir / "features/timestamps.csv")
    out_fold = out / fold
    out_fold.mkdir(parents=True, exist_ok=True)

    # ---- cells (full + focus) ----
    base_cols = ["t_idx", "datetime", "node_idx", "y_true", "z_true", "z_mask"]
    model_cols: list[str] = []
    for key in (k for k, _, _ in MODELS):
        model_cols += [f"{key}_score", f"{key}_alarm", f"{key}_conf"]

    focus_rows: list[dict] = []
    full_path = out_fold / "cells_full.csv.gz"
    full_fh = gzip.open(full_path, "wt", newline="") if write_full else None
    full_writer = None
    if full_fh is not None:
        full_writer = csv.DictWriter(full_fh, fieldnames=base_cols + model_cols)
        full_writer.writeheader()

    for r, c in zip(ri.tolist(), ci.tolist()):
        t_idx = int(test_t[r])
        zt = int(z_t[r, c])
        row = {
            "t_idx": t_idx,
            "datetime": timestamps[t_idx],
            "node_idx": int(c),
            "y_true": int(y_t[r, c]),
            "z_true": zt,
            "z_mask": int(zmask_t[r, c]),
        }
        any_alarm = False
        for key in (k for k, _, _ in MODELS):
            sval = scores[key][r, c]
            if np.isnan(sval):
                score = None
                alarm = None
            else:
                score = round(float(sval), 6)
                tau = taus[key]
                alarm = bool(score >= tau) if tau is not None else None
                any_alarm = any_alarm or bool(alarm)
            row[f"{key}_score"] = "" if score is None else score
            row[f"{key}_alarm"] = "" if alarm is None else int(alarm)
            row[f"{key}_conf"] = confusion(score, alarm, zt)
        if full_writer is not None:
            full_writer.writerow(row)
        if zt == 1 or any_alarm:
            focus_rows.append(row)

    if full_fh is not None:
        full_fh.close()

    (out_fold / "cells_focus.json").write_text(
        json.dumps(focus_rows, ensure_ascii=False), encoding="utf-8"
    )

    # ---- timeline: per-timestamp aggregates ----
    timeline = []
    for r in range(test_t.size):
        t_idx = int(test_t[r])
        zr = z_t[r]
        entry = {"t_idx": t_idx, "datetime": timestamps[t_idx], "n_onset": int((zr == 1).sum())}
        for key in (k for k, _, _ in MODELS):
            tau = taus[key]
            sr = scores[key][r]
            valid = ~np.isnan(sr)
            if tau is None or not valid.any():
                entry[key] = {"n_alarm": 0, "tp": 0, "fp": 0, "fn": int((zr == 1).sum())}
                continue
            alarm = valid & (sr >= tau)
            tp = int((alarm & (zr == 1)).sum())
            fp = int((alarm & (zr == 0)).sum())
            fn = int(((zr == 1) & ~alarm).sum())
            entry[key] = {"n_alarm": int(alarm.sum()), "tp": tp, "fp": fp, "fn": fn}
        # only keep timestamps with any activity to keep the file small
        if entry["n_onset"] > 0 or any(entry[k]["n_alarm"] > 0 for k, _, _ in MODELS):
            timeline.append(entry)
    (out_fold / "timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False), encoding="utf-8"
    )

    meta = {
        "fold": fold,
        "percentile": percentile,
        "highlight_model": HIGHLIGHT_MODEL,
        "models": [k for k, _, _ in MODELS],
        "models_present": present,
        "tau": {k: taus[k] for k, _, _ in MODELS},
        "counts": {
            "test_timestamps": int(test_t.size),
            "universe_cells": int(ri.size),
            "focus_cells": len(focus_rows),
            "onset_cells": int((z_t == 1).sum()),
            "timeline_points": len(timeline),
        },
        "confusion_legend": {
            "TP": "z=1 & alarm", "FP": "z=0 & alarm",
            "FN": "z=1 & no alarm", "TN": "z=0 & no alarm", "nodata": "no prediction",
        },
    }
    (out_fold / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{fold}] focus={len(focus_rows)} timeline={len(timeline)} -> {out_fold}", flush=True)
    return meta


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=repo / "data/outputs/run_all")
    ap.add_argument("--data-dir", type=Path, default=repo / "data/train_ready/d7_active_giant_full")
    ap.add_argument(
        "--nodes-meta",
        type=Path,
        default=repo / "data/train_ready/d7_active_giant_2016_01/graph/nodes.csv",
        help="CSV with segment_id + centroid_lon/lat/route/length_m (geometry source).",
    )
    ap.add_argument("--out", type=Path, default=repo / "data/outputs/case_study")
    ap.add_argument("--percentile", type=str, default="p97")
    ap.add_argument("--folds", nargs="+", default=None, help="default: all folds found")
    ap.add_argument("--no-full", action="store_true", help="skip the large cells_full.csv.gz")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    nodes = load_node_meta(args.data_dir, args.nodes_meta)
    (args.out / "nodes.json").write_text(json.dumps(nodes, ensure_ascii=False), encoding="utf-8")
    print(f"[nodes] {len(nodes)} nodes -> {args.out / 'nodes.json'}", flush=True)

    from flood_traffic.constants import FOLDS

    folds = args.folds or FOLDS
    for fold in folds:
        export_fold(args.base, args.data_dir, args.percentile, fold, args.out, write_full=not args.no_full)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
