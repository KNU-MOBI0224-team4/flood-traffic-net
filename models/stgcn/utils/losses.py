import torch
import torch.nn as nn


class MaskedBCEWithLogitsLoss(nn.Module):

    def __init__(self, pos_weight=None):
        super().__init__()

        self.criterion = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=pos_weight,
        )

    def forward(self, logits, targets, mask):

        """
        logits: [B, N]
        targets: [B, N]
        mask: [B, N]
        """

        # ---------------------------------
        # Element-wise BCE
        # ---------------------------------
        loss = self.criterion(logits, targets)

        # ---------------------------------
        # Apply mask
        # ---------------------------------
        loss = loss * mask.float()

        # ---------------------------------
        # Average only valid positions
        # ---------------------------------
        loss = loss.sum() / (mask.float().sum() + 1e-8)

        return loss