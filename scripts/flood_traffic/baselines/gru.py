"""GRU baseline."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from flood_traffic.metrics import safe_metric
from flood_traffic.sequence_data import GRUSplit


MODEL_NAME = "gru"


class GRUClassifier(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output, hidden = self.gru(x)

        last_hidden = hidden[-1]

        logits = self.fc(last_hidden)

        return logits.squeeze(-1)


def fit(
    train_split: GRUSplit,
    val_split: GRUSplit,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 10,
    seed: int = 42,
    device: str = "cpu",
    pos_weight_cap: float = 50.0,
    grad_clip: float = 5.0,
    early_stopping_patience: int = 0,
) -> tuple[Any, dict[str, Any]]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = torch.tensor(train_split.X, dtype=torch.float32)
    y_train = torch.tensor(train_split.y, dtype=torch.float32)
    y_val = val_split.y.astype(np.float64)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    model = GRUClassifier(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Class imbalance: onset cells are ~0.01-0.04% of node-hours, so weight the
    # positive class (capped) the same way the STGCN trainer does.
    pos = float(y_train.sum())
    neg = float(len(y_train)) - pos
    pos_weight_value = min(neg / (pos + 1e-8), pos_weight_cap)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auprc = -math.inf
    best_state_dict = None
    history: list[dict[str, Any]] = []
    no_improve = 0
    stopped_at = None

    for epoch in range(epochs):

        start = time.time()
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()

        # Batched inference so full-period val splits do not OOM in one pass.
        scores = predict(model, val_split.X, batch_size=batch_size, device=device)
        val_auprc = safe_metric(average_precision_score, y_val, scores)

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(len(train_loader), 1),
            "val_auprc": float(val_auprc),
            "elapsed_sec": time.time() - start,
        }
        history.append(epoch_info)
        print(
            f"epoch={epoch+1} "
            f"train_loss={epoch_info['train_loss']:.6f} "
            f"val_auprc={val_auprc:.6f}",
            flush=True,
        )

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if early_stopping_patience > 0 and no_improve >= early_stopping_patience:
            stopped_at = epoch + 1
            break

    if best_state_dict is None:
        raise RuntimeError("Failed to train GRU baseline")

    model.load_state_dict(best_state_dict)

    info = {
        "best_val_auprc": float(best_val_auprc),
        "epochs": int(epochs),
        "stopped_at_epoch": stopped_at,
        "early_stopping_patience": int(early_stopping_patience),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "pos_weight": float(pos_weight_value),
        "pos_weight_cap": float(pos_weight_cap),
        "grad_clip": float(grad_clip),
        "history": history,
    }

    return model, info


def predict(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 512,
    device: str | None = None,
) -> np.ndarray:

    # The shared evaluator calls predict(model, X) with no device, so follow the
    # device the model already lives on instead of forcing CPU (which would
    # mismatch a CUDA-trained model).
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.shape[0] == 0:
        return np.array([], dtype=np.float64)

    X_tensor = torch.tensor(X_arr, dtype=torch.float32)
    loader = DataLoader(X_tensor, batch_size=batch_size, shuffle=False)

    outputs = []
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            scores = torch.sigmoid(logits)
            outputs.append(np.asarray(scores.cpu().numpy(), dtype=np.float64).reshape(-1))

    if len(outputs) == 0:
        return np.array([], dtype=np.float64)
    return np.concatenate(outputs).astype(np.float64)

