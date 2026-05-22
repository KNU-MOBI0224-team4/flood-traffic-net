# Flood Traffic Net

Flood-induced road-paralysis onset prediction experiments on the D7 road graph.

The repository keeps a compact January 2016 sample for onboarding and code
validation. Full-period train-ready tensors and model outputs are generated
locally and are intentionally not pushed to GitHub.

## Repository Layout

```text
data/
  data_sample/                         # tracked source-style January 2016 sample
  data_train/
    d7_active_giant_2016_01/           # tracked train-ready January 2016 sample
    d7_active_giant_full/              # local full-period data, gitignored
  outputs/                             # local model outputs, gitignored
scripts/
  build_train_ready_sample.py
  build_train_ready_full.py
  run_tabular_baselines.py
  flood_traffic/
    baselines/
    metrics.py
    sampling.py
    tabular_data.py
    evaluation.py
```

## Data Policy

- `data/data_sample/` is a small source-style sample and should stay tracked.
- `data/data_train/d7_active_giant_2016_01/` is a small train-ready sample and should stay tracked.
- `data/data_train/d7_active_giant_full/` is the full-period training dataset and is gitignored.
- `data/outputs/` contains experiment outputs and is gitignored.

## Quick Start

Run all commands from the repository root.

1. Create the Python environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r data/requirements-baselines.txt
```

2. Optional: regenerate the tracked January 2016 train-ready sample:

```bash
python3 scripts/build_train_ready_sample.py
```

3. Build the full-period train-ready dataset:

```bash
python3 scripts/build_train_ready_full.py
```

This writes local arrays to `data/data_train/d7_active_giant_full/`.

If the processed research data is not in the default sibling directory
`../Flood_induced_road_paralysis`, pass it explicitly:

```bash
python3 scripts/build_train_ready_full.py \
  --project-root /path/to/Flood_induced_road_paralysis
```

4. Run tabular baselines:

```bash
python3 scripts/run_tabular_baselines.py \
  --data-dir data/data_train/d7_active_giant_full \
  --out-dir data/outputs/tabular_baselines \
  --percentiles p99 p97 \
  --models logistic xgboost
```

5. Check outputs:

```text
data/outputs/tabular_baselines/
  metrics_summary.csv
  run_config.json
  p97/{fold}/...
  p99/{fold}/...
```

For a quick smoke test, run only one fold and one XGBoost boosting round:

```bash
python3 scripts/run_tabular_baselines.py \
  --data-dir data/data_train/d7_active_giant_full \
  --out-dir data/outputs/tabular_baselines_smoke \
  --percentiles p99 \
  --folds fold_1_train2016_2020_val2021_test2022 \
  --models xgboost \
  --xgb-rounds 1 \
  --xgb-early-stopping-rounds 1
```

## Baseline Definition

The current tabular baselines use:

- Dynamic features: `rainfall_mm_1h`, `avg_speed_median`, `total_flow_median`, `avg_occupancy_median`, and 1-hour deltas.
- Static feature: `node_z_history_count_norm`.
- Target: `z`, evaluated only where `z_mask == 1`.
- Train sampling: all positive-onset timestamps plus sampled negative-only timestamps.
- Metrics: AUPRC, Recall@Precision>=0.10, threshold precision/recall/F1, event-level recall, and diagnostic y=1 alarm rate.

## Baseline Code Structure

- `scripts/flood_traffic/baselines/logistic_regression.py`: Logistic Regression baseline.
- `scripts/flood_traffic/baselines/xgboost_model.py`: XGBoost baseline.
- `scripts/flood_traffic/metrics.py`: shared metrics and event-level recall.
- `scripts/flood_traffic/sampling.py`: shared train timestamp sampling.
- `scripts/flood_traffic/tabular_data.py`: train-ready data loading and tabular row construction.
- `scripts/flood_traffic/evaluation.py`: shared validation/test evaluation loop.
