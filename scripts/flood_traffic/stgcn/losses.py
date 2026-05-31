"""Masked losses for onset prediction with z_mask."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MaskedFocalLoss(nn.Module):
    """Sigmoid Focal Loss averaged over positions where mask == 1.

    Lin et al. (2017) "Focal Loss for Dense Object Detection".
    L = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t = p if y=1 else (1-p), alpha_t = alpha if y=1 else (1-alpha).

    Compared to BCE+pos_weight:
    - (1-p_t)^gamma 가 이미 잘 맞히는 cell의 loss를 down-weight → 쉬운 negative
      99%가 학습 신호를 압도하는 문제를 자연 완화.
    - alpha 는 양성/음성 베이스 가중치 (pos_weight과 비슷하지만 곱셈이 아니라
      가산 방식이라 gradient 폭주 위험이 낮음).

    logits/targets/mask all share shape (B, N).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        loss = loss * mask.float()
        return loss.sum() / (mask.float().sum() + 1e-8)
