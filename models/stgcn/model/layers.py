import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    """
    Temporal Convolution Layer

    Input:
        [B, T, N, C]

    Output:
        [B, T, N, C_out]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
        )

    def forward(self, x):

        # ---------------------------------
        # [B, T, N, C]
        # -> [B, C, T, N]
        # ---------------------------------
        x = x.permute(0, 3, 1, 2)

        # ---------------------------------
        # Temporal convolution
        # ---------------------------------
        x = self.conv(x)

        x = F.relu(x)

        # ---------------------------------
        # [B, C, T, N]
        # -> [B, T, N, C]
        # ---------------------------------
        x = x.permute(0, 2, 3, 1)

        return x


class GraphConv(nn.Module):
    """
    Simple Graph Convolution Layer

    Input:
        x: [B, T, N, C]
        A_hat: [N, N]

    Output:
        [B, T, N, C_out]
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, A_hat):

        # ---------------------------------
        # Graph propagation
        # ---------------------------------
        x = torch.einsum(
            "ij,btjf->btif",
            A_hat,
            x
        )

        # ---------------------------------
        # Feature transform
        # ---------------------------------
        x = self.linear(x)

        x = F.relu(x)

        return x