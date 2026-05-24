"""Masked BCE loss for onset prediction with z_mask."""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE-with-logits averaged over positions where mask == 1.

    logits/targets/mask all share shape (B, N).
    """

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.criterion(logits, targets)
        loss = loss * mask.float()
        return loss.sum() / (mask.float().sum() + 1e-8)
