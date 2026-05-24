"""Temporal and graph convolution layers used by STGCN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    """1D temporal convolution applied per node.

    Input/output shape: (B, T, N, C).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)
        return x


class GraphConv(nn.Module):
    """Single-step graph convolution with a precomputed normalized adjacency.

    Input shape: (B, T, N, C_in), A_hat shape: (N, N).
    Output shape: (B, T, N, C_out).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("ij,btjf->btif", A_hat, x)
        x = self.linear(x)
        x = F.relu(x)
        return x
