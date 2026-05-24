"""STGCN model: two ST-Conv blocks + temporal conv + per-node prediction head."""

from __future__ import annotations

import torch
import torch.nn as nn

from flood_traffic.stgcn.layers import TemporalConv
from flood_traffic.stgcn.stgcn_block import STConvBlock


class STGCN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.block1 = STConvBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.block2 = STConvBlock(
            in_channels=hidden_channels,
            hidden_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.final_temp = TemporalConv(out_channels, out_channels, kernel_size)
        self.fc = nn.Linear(out_channels, 1)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        x = self.block1(x, A_hat)
        x = self.block2(x, A_hat)
        x = self.final_temp(x)
        x = x[:, -1]
        x = self.fc(x)
        x = x.squeeze(-1)
        return x
