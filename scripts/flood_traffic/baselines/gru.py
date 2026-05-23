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

        return logits.squeeze(1)


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
) -> tuple[Any, dict[str, Any]]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = torch.tensor(train_split.X, dtype=torch.float32)
    y_train = torch.tensor(train_split.y, dtype=torch.float32)

    X_val = torch.tensor(val_split.X, dtype=torch.float32)
    y_val = torch.tensor(val_split.y, dtype=torch.float32)

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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_auprc = -math.inf
    best_state_dict = None

    history = []

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

            optimizer.step()

            total_loss += loss.item()

        model.eval()

        with torch.no_grad():

            logits = model(X_val.to(device))

            scores = torch.sigmoid(logits).cpu().numpy()

        val_auprc = safe_metric(
            average_precision_score,
            y_val.numpy(),
            scores,
        )

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_loader),
            "val_auprc": val_auprc,
            "elapsed_sec": time.time() - start,
        }

        history.append(epoch_info)
        print(
            f"epoch={epoch+1} "
            f"train_loss={total_loss/len(train_loader):.6f} "
            f"val_auprc={val_auprc:.6f}"
            )



        if val_auprc > best_val_auprc:

            best_val_auprc = val_auprc

            best_state_dict = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state_dict is None:
        raise RuntimeError("Failed to train GRU baseline")

    model.load_state_dict(best_state_dict)

    info = {
        "best_val_auprc": best_val_auprc,
        "history": history,
    }

    return model, info


def predict(
    model: Any,
    X: np.ndarray,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:

    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)

    loader = DataLoader(
        X_tensor,
        batch_size=batch_size,
        shuffle=False,
    )

    outputs = []

    with torch.no_grad():

        for batch_x in loader:

            batch_x = batch_x.to(device)

            logits = model(batch_x)

            scores = torch.sigmoid(logits)

            outputs.append(scores.cpu().numpy())

    return np.concatenate(outputs).astype(np.float64)

