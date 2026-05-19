# Train-Ready Data

This directory contains model-ready arrays generated from `data_sample/`.

Current package:

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

## Arrays

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

Run this from the repository root:

```bash
python3 scripts/build_train_ready_sample.py
```

The script uses only the Python standard library and writes NumPy-compatible
`.npz` files.
