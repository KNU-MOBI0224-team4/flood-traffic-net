import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from models.stgcn.data.dataset import FloodSTGCNDataset
from models.stgcn.data.graph_loader import load_adjacency_matrix

from models.stgcn.model.stgcn import STGCN

from models.stgcn.utils.losses import (
    MaskedBCEWithLogitsLoss,
)

from models.stgcn.utils.trainer import (
    train_one_epoch,
    evaluate,
)


# =====================================================
# CONFIG
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEQ_LEN = 12
BATCH_SIZE = 2
EPOCHS = 1
LR = 1e-4

# =====================================================
# PATHS
# =====================================================

ROOT = r"C:\Users\graph\Desktop\flood-traffic-net\data\data_train\d7_active_giant_full"

X_PATH = os.path.join(
    ROOT,
    "features",
    "X_dynamic.npz",
)

TIMESTAMP_PATH = os.path.join(
    ROOT,
    "features",
    "timestamps.csv",
)

TARGET_PATH = os.path.join(
    ROOT,
    "labels",
    "p97",
    "fold_2_train2016_2021_val2022_test2023",
    "targets.npz",
)

GRAPH_PATH = os.path.join(
    ROOT,
    "graph",
    "adjacency_matrix.csv",
)

# =====================================================
# DATASET
# =====================================================

train_dataset = FloodSTGCNDataset(
    x_path=X_PATH,
    target_path=TARGET_PATH,
    timestamp_path=TIMESTAMP_PATH,
    seq_len=SEQ_LEN,
    split="train",
)

val_dataset = FloodSTGCNDataset(
    x_path=X_PATH,
    target_path=TARGET_PATH,
    timestamp_path=TIMESTAMP_PATH,
    seq_len=SEQ_LEN,
    split="val",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# =====================================================
# GRAPH
# =====================================================

A_hat = load_adjacency_matrix(
    GRAPH_PATH
).to(DEVICE)

# =====================================================
# POSITIVE WEIGHT
# =====================================================

all_targets = train_dataset.z
all_masks = train_dataset.z_mask

valid_targets = all_targets[all_masks]

pos_count = valid_targets.sum()
neg_count = len(valid_targets) - pos_count

raw_pos_weight = neg_count / (pos_count + 1e-8)

print(f"Raw Positive weight: {raw_pos_weight:.4f}")

pos_weight = torch.tensor(
    min(raw_pos_weight, 50.0),
    dtype=torch.float32,
).to(DEVICE)

print(f"Capped Positive weight: {pos_weight.item():.4f}")

print(f"Positive weight: {pos_weight.item():.4f}")

# =====================================================
# MODEL
# =====================================================

model = STGCN(
    num_nodes=329,
    in_channels=7,
    hidden_channels=32,
    out_channels=64,
).to(DEVICE)




# =====================================================
# LOSS
# =====================================================

criterion = MaskedBCEWithLogitsLoss(
    pos_weight=pos_weight
)

# =====================================================
# OPTIMIZER
# =====================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
)

# =====================================================
# TRAIN LOOP
# =====================================================

best_pr_auc = 0

for epoch in range(EPOCHS):

    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    # ---------------------------------
    # Train
    # ---------------------------------
    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        A_hat,
        DEVICE,
    )

    print(f"Train Loss: {train_loss:.6f}")

    # ---------------------------------
    # Validation
    # ---------------------------------
    metrics = evaluate(
        model,
        val_loader,
        criterion,
        A_hat,
        DEVICE,
    )

    print(
        f"Val Loss: {metrics['loss']:.6f} | "
        f"PR-AUC: {metrics['pr_auc']:.6f} | "
        f"ROC-AUC: {metrics['roc_auc']:.6f} | "
        f"F1: {metrics['f1']:.6f} | "
        f"Recall: {metrics['recall']:.6f}"
    )

    # ---------------------------------
    # Save best model
    # ---------------------------------
    if metrics["pr_auc"] > best_pr_auc:

        best_pr_auc = metrics["pr_auc"]

        torch.save(
            model.state_dict(),
            "best_stgcn.pth"
        )

        print("✅ Best model saved!")