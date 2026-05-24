# Data Raw

This directory contains source-style D7 sample files for model development and
team onboarding. These files are useful for inspecting the original tabular
format before it is converted into train-ready tensors under `train_ready/`.

This sample is intentionally tracked in GitHub. Full-period generated tensors
and model outputs are not tracked.

## Layout

```text
raw/d7_active_giant_2016_01/
  graph/d7_active_giant/
    D7_active_giant_nodes.csv
    D7_active_giant_adjacency_edges.csv
    D7_active_giant_adjacency_matrix.csv
  samples/d7_active_giant/2016_01/
    traffic/D7_active_giant_node_hourly_2016_01.csv.gz
    rainfall/D7_active_giant_node_rainfall_2016_01.csv.gz
    labels/{p97,p99}/{fold}/D7_active_giant_labels_{p97,p99}_2016_01.csv.gz
  static/d7_active_giant/z_history/{p97,p99}/{fold}/
    static_node_features_z_history.csv
  metadata/d7_active_giant/
    label_thresholds/{p97,p99}/{fold}/
    label_summaries/{p97,p99}/{fold}/
  summaries/d7_active_giant/
    D7_rolling_3fold_label_summary.csv
    static_node_features_z_history_all_summary.json
```

## Graph

- `D7_active_giant_nodes.csv`: 329 road-segment nodes retained in the active
  giant component.
- `D7_active_giant_adjacency_edges.csv`: 774 undirected road-connectivity edges.
- `D7_active_giant_adjacency_matrix.csv`: 329 x 329 adjacency matrix using
  `segment_id` as both row and column identifiers.

## January 2016 Samples

- `traffic`: hourly node-level PeMS traffic aggregation.
- `rainfall`: hourly node-level GaugeCorr QPE rainfall matched to each road node.
- `labels`: p97 and p99 flood-seed percentile labels for each rolling fold.

The primary dynamic model inputs are:

- `rainfall_mm_1h`
- `avg_speed_median`
- `total_flow_median`
- `avg_occupancy_median`

The supervised target is `z`, evaluated and trained only where `z_mask == 1`.
The state label `y` is kept for event interpretation and for constructing the
onset target, but it should not be used directly as the training target for the
current onset-prediction task.

The train-ready version of this sample is generated with:

```bash
python3 scripts/build_train_ready_sample.py
```

from the repository root.

## Rolling Folds

- `fold_1_train2016_2020_val2021_test2022`
- `fold_2_train2016_2021_val2022_test2023`
- `fold_3_train2016_2022_val2023_test2024`

Static `node_z_history` features are fold-specific because they are computed
from each fold's train period only.
