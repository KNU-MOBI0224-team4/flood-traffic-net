"""ST-Conv block: temporal -> graph -> temporal -> dropout."""

from __future__ import annotations

import torch
import torch.nn as nn

from flood_traffic.stgcn.layers import GraphConv, TemporalConv


class STConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.3,
        static_dim: int = 0,
        static_embedding_dim: int = 8,
        cheb_k: int = 3,
        hidden_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.temp_conv1 = TemporalConv(in_channels, hidden_channels, kernel_size)
        self.norm1 = nn.LayerNorm(hidden_channels) if hidden_layernorm else nn.Identity()
        self.graph_conv = GraphConv(
            hidden_channels,
            hidden_channels,
            static_dim=static_dim,
            static_embedding_dim=static_embedding_dim,
            cheb_k=cheb_k,
        )
        self.norm2 = nn.LayerNorm(hidden_channels) if hidden_layernorm else nn.Identity()
        self.temp_conv2 = TemporalConv(hidden_channels, out_channels, kernel_size)
        self.norm3 = nn.LayerNorm(out_channels) if hidden_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        A_hat: torch.Tensor,
        static_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.temp_conv1(x)
        x = self.norm1(x)
        x = self.graph_conv(x, A_hat, static_features)
        x = self.norm2(x)
        x = self.temp_conv2(x)
        x = self.norm3(x)
        x = self.dropout(x)
        return x
