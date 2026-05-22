"""Shared constants for D7 flood-traffic experiments."""

PERCENTILES = ["p99", "p97"]

FOLDS = [
    "fold_1_train2016_2020_val2021_test2022",
    "fold_2_train2016_2021_val2022_test2023",
    "fold_3_train2016_2022_val2023_test2024",
]

SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2, "unused": 3}

STATIC_FEATURES = ["node_z_history_count_norm"]

