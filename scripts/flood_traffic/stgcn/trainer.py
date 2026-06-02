"""STGCN training loop and inference helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from flood_traffic.graph_data import STGCNDataset
from flood_traffic.metrics import safe_metric
from flood_traffic.stgcn.losses import MaskedBCEWithLogitsLoss, MaskedFocalLoss
from flood_traffic.stgcn.stgcn import STGCN


MODEL_NAME = "stgcn"


def _make_loader(dataset: STGCNDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def _infer_loader(
    model: STGCN,
    loader: DataLoader,
    A_hat: torch.Tensor,
    static_features: torch.Tensor | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return concatenated (scores, targets, masks) of shape (num_samples, N)."""
    model.eval()
    scores, targets, masks = [], [], []
    for x, y, m, _t in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x, A_hat, static_features)
        prob = torch.sigmoid(logits)
        scores.append(prob.cpu().numpy())
        targets.append(y.cpu().numpy())
        masks.append(m.cpu().numpy())
    return (
        np.concatenate(scores, axis=0).astype(np.float64),
        np.concatenate(targets, axis=0).astype(np.float64),
        np.concatenate(masks, axis=0).astype(bool),
    )


def _train_one_epoch(
    model: STGCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MaskedBCEWithLogitsLoss,
    A_hat: torch.Tensor,
    static_features: torch.Tensor | None,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y, m, _t in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        logits = model(x, A_hat, static_features)
        loss = criterion(logits, y, m)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.detach())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _compute_pos_weight(
    dataset: STGCNDataset,
    cap: float,
    device: torch.device,
) -> torch.Tensor:
    z_sel = dataset.z[dataset.target_timestamps]
    m_sel = dataset.z_mask[dataset.target_timestamps]
    valid = z_sel[m_sel]
    pos = float(valid.sum())
    neg = float(len(valid)) - pos
    raw = neg / (pos + 1e-8)
    return torch.tensor(min(raw, cap), dtype=torch.float32, device=device)


def fit(
    train_dataset: STGCNDataset,
    val_dataset: STGCNDataset,
    A_hat: np.ndarray,
    static_features: np.ndarray | None,
    *,
    num_nodes: int,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    kernel_size: int,
    dropout: float,
    static_embedding_dim: int,
    cheb_k: int,
    hidden_layernorm: bool,
    epochs: int,
    lr: float,
    batch_size: int,
    pos_weight_cap: float,
    grad_clip: float,
    device: str,
    seed: int,
    early_stopping_patience: int = 0,
    loss_type: str = "bce",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> tuple[STGCN, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_obj = torch.device(device)
    A_hat_t = torch.from_numpy(A_hat).float().to(device_obj)
    static_t: torch.Tensor | None = None
    static_dim = 0
    if static_features is not None and static_features.shape[1] > 0:
        static_t = torch.from_numpy(static_features).float().to(device_obj)
        static_dim = int(static_t.shape[1])

    pos_weight = _compute_pos_weight(train_dataset, pos_weight_cap, device_obj)

    model = STGCN(
        num_nodes=num_nodes,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        static_dim=static_dim,
        static_embedding_dim=static_embedding_dim,
        cheb_k=cheb_k,
        hidden_layernorm=hidden_layernorm,
    ).to(device_obj)

    if loss_type == "focal":
        criterion = MaskedFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == "bce":
        criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = _make_loader(train_dataset, batch_size, shuffle=True)
    val_loader = _make_loader(val_dataset, batch_size, shuffle=False)

    best_val_auprc = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    no_improve = 0
    stopped_at: int | None = None

    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, criterion, A_hat_t, static_t, device_obj, grad_clip
        )
        val_scores, val_targets, val_masks = _infer_loader(
            model, val_loader, A_hat_t, static_t, device_obj
        )
        flat_scores = val_scores[val_masks]
        flat_targets = val_targets[val_masks]
        val_auprc = safe_metric(average_precision_score, flat_targets, flat_scores)

        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_auprc": float(val_auprc),
            "elapsed_sec": time.time() - epoch_start,
        }
        if val_auprc > best_val_auprc:
            best_val_auprc = float(val_auprc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            record["is_best"] = True
            no_improve = 0
        else:
            no_improve += 1
        history.append(record)
        if early_stopping_patience > 0 and no_improve >= early_stopping_patience:
            stopped_at = epoch + 1
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    info: dict[str, Any] = {
        "epochs": int(epochs),
        "early_stopping_patience": int(early_stopping_patience),
        "stopped_at_epoch": stopped_at,
        "lr": float(lr),
        "batch_size": int(batch_size),
        "loss_type": loss_type,
        "temporal_conv": "glu",
        "static_injection": "gcn_input_initialization" if static_dim > 0 else "disabled",
        "static_dim": int(static_dim),
        "static_embedding_dim": int(static_embedding_dim) if static_dim > 0 else None,
        "graph_conv_type": "cheb",
        "cheb_k": int(cheb_k),
        "hidden_layernorm": bool(hidden_layernorm),
        "focal_alpha": float(focal_alpha) if loss_type == "focal" else None,
        "focal_gamma": float(focal_gamma) if loss_type == "focal" else None,
        "pos_weight": float(pos_weight.item()),
        "pos_weight_cap": float(pos_weight_cap),
        "grad_clip": float(grad_clip),
        "best_val_auprc": float(best_val_auprc),
        "history": history,
    }
    return model, info


@torch.no_grad()
def predict_score_matrix(
    model: STGCN,
    dataset: STGCNDataset,
    A_hat: np.ndarray,
    static_features: np.ndarray | None,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (target_timestamps, scores) where scores has shape (num_targets, N)."""
    device_obj = torch.device(device)
    A_hat_t = torch.from_numpy(A_hat).float().to(device_obj)
    static_t: torch.Tensor | None = None
    if static_features is not None and static_features.shape[1] > 0:
        static_t = torch.from_numpy(static_features).float().to(device_obj)
    loader = _make_loader(dataset, batch_size, shuffle=False)
    scores, _t, _m = _infer_loader(model.to(device_obj), loader, A_hat_t, static_t, device_obj)
    return dataset.target_timestamps.copy(), scores


def save_model(model: STGCN, path: Path) -> None:
    torch.save(model.state_dict(), str(path))
