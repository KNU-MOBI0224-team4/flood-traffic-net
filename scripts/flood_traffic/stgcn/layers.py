"""Temporal and graph convolution layers used by STGCN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    """GLU temporal convolution applied per node.

    Input/output shape: (B, T, N, C).

    The convolution output is split into signal/gate channels:
        output = P * sigmoid(Q)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * 2,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        signal, gate = torch.chunk(x, chunks=2, dim=1)
        x = signal * torch.sigmoid(gate)
        x = x.permute(0, 2, 3, 1)
        return x


class GraphConv(nn.Module):
    """Chebyshev graph convolution with static node-feature initialization.

    Input shape: (B, T, N, C_in), A_hat shape: (N, N).
    Output shape: (B, T, N, C_out).

    Uses K Chebyshev supports:
        [T_0(X), T_1(X), ..., T_{K-1}(X)]
    where T_0(X)=X, T_1(X)=A_hat X, and
    T_k(X)=2 A_hat T_{k-1}(X)-T_{k-2}(X).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        static_dim: int = 0,
        static_embedding_dim: int = 8,
        cheb_k: int = 3,
    ) -> None:
        super().__init__()
        self.static_dim = int(static_dim)
        self.static_embedding_dim = int(static_embedding_dim)
        self.cheb_k = int(cheb_k)
        if self.cheb_k < 1:
            raise ValueError("cheb_k must be >= 1")
        if self.static_dim > 0:
            self.static_encoder = nn.Linear(self.static_dim, self.static_embedding_dim)
            self.static_projection = nn.Linear(
                in_channels + self.static_embedding_dim, in_channels
            )
        else:
            self.static_encoder = None
            self.static_projection = None
        self.linear = nn.Linear(in_channels * self.cheb_k, out_channels)

    @staticmethod
    def _propagate(A_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,btjf->btif", A_hat, x)

    def _chebyshev_stack(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        supports = [x]
        if self.cheb_k == 1:
            return supports[0]
        supports.append(self._propagate(A_hat, x))
        for _ in range(2, self.cheb_k):
            supports.append(2.0 * self._propagate(A_hat, supports[-1]) - supports[-2])
        return torch.cat(supports, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        A_hat: torch.Tensor,
        static_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.static_dim > 0:
            if static_features is None:
                raise ValueError("static_features must be provided when static_dim > 0")
            if static_features.shape[0] != x.shape[2]:
                raise ValueError(
                    f"static node count {static_features.shape[0]} does not match x node count {x.shape[2]}"
                )
            static_emb = self.static_encoder(static_features)
            static_emb = static_emb[None, None, :, :].expand(
                x.shape[0], x.shape[1], -1, -1
            )
            x = torch.cat([x, static_emb], dim=-1)
            x = self.static_projection(x)

        x = self._chebyshev_stack(x, A_hat)
        x = self.linear(x)
        x = F.relu(x)
        return x
