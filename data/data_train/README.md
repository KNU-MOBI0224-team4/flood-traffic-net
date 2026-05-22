# Train-Ready Data

This directory contains model-ready arrays. The compact January 2016 sample is
tracked in GitHub; the full-period training package is generated locally and
ignored by Git.

Tracked sample package:

```text
data_train/
  d7_active_giant_2016_01/
    graph/
      A.npz
      adjacency_matrix.csv
      edge_index.csv
      edges.csv
      graph_spec.json
      nodes.csv
    features/
      X_dynamic.npz
      feature_spec.json
      node_ids.csv
      timestamps.csv
    labels/{p97,p99}/{fold}/
      targets.npz
      X_static.csv
      label_node_thresholds.csv
      label_summary.json
      target_spec.json
    summaries/
      D7_rolling_3fold_label_summary.csv
      static_node_features_z_history_all_summary.json
    manifest.json
```

Local full-period package:

```text
data_train/
  d7_active_giant_full/
    graph/
      A.npy
      edge_index.csv
      graph_spec.json
      node_ids.csv
    features/
      X_dynamic.npy
      input_mask.npy
      continuous_prev_hour.npy
      feature_spec.json
      timestamps.csv
    labels/{p97,p99}/{fold}/
      y.npy
      z.npy
      z_mask.npy
      label_available.npy
      split_code.npy
      X_static.csv
      label_node_thresholds.csv
      label_summary.json
      target_spec.json
    manifest.json
```

## Arrays

For the tracked January 2016 sample:

- `features/X_dynamic.npz`
  - `X_dynamic`: shape `(T, N, F) = (744, 329, 7)`
  - `input_mask`: same shape, where `1` means the feature value is observed
- `graph/A.npz`
  - `A`: shape `(N, N)`, binary road-connectivity adjacency matrix
- `labels/{p97,p99}/{fold}/targets.npz`
  - `y`: flood-induced road-paralysis state label
  - `z`: onset target label
  - `z_mask`: loss/evaluation mask for `z`
  - `label_available`: label availability mask

## Dynamic Features

The dynamic feature order is stored in `features/feature_spec.json`.

```text
0. rainfall_mm_1h
1. avg_speed_median
2. total_flow_median
3. avg_occupancy_median
4. avg_speed_median_delta_1h
5. total_flow_median_delta_1h
6. avg_occupancy_median_delta_1h
```

The delta features are defined as `value[t] - value[t-1]`, so they use only
past/current information at anchor time `t` and do not leak future information.

## Regeneration

Regenerate the tracked January 2016 sample from the repository root:

```bash
python3 scripts/build_train_ready_sample.py
```

The script uses only the Python standard library and writes NumPy-compatible
`.npz` files.

Build the full-period local package:

```bash
python3 scripts/build_train_ready_full.py
```

The full-period package is written to `data/data_train/d7_active_giant_full/`
and is excluded from GitHub by `.gitignore`.

## Baseline Training

Tabular baselines consume the full-period package:

```bash
python3 scripts/run_tabular_baselines.py \
  --data-dir data/data_train/d7_active_giant_full \
  --out-dir data/outputs/tabular_baselines \
  --percentiles p99 p97 \
  --models logistic xgboost
```

The shared metric and timestamp sampling implementations live under
`scripts/flood_traffic/` so future baselines and graph/sequence models can
reuse the same evaluation protocol.
