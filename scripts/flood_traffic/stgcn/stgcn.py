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
        static_dim: int = 0,
        static_embedding_dim: int = 8,
        cheb_k: int = 3,
        hidden_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.block1 = STConvBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            static_dim=static_dim,
            static_embedding_dim=static_embedding_dim,
            cheb_k=cheb_k,
            hidden_layernorm=hidden_layernorm,
        )
        self.block2 = STConvBlock(
            in_channels=hidden_channels,
            hidden_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            static_dim=static_dim,
            static_embedding_dim=static_embedding_dim,
            cheb_k=cheb_k,
            hidden_layernorm=hidden_layernorm,
        )
        self.final_temp = TemporalConv(out_channels, out_channels, kernel_size)
        self.final_norm = nn.LayerNorm(out_channels) if hidden_layernorm else nn.Identity()
        self.fc = nn.Linear(out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        A_hat: torch.Tensor,
        static_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.block1(x, A_hat, static_features)
        x = self.block2(x, A_hat, static_features)
        x = self.final_temp(x)
        x = self.final_norm(x)
        x = x[:, -1]
        x = self.fc(x)
        x = x.squeeze(-1)
        return x
