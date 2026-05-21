import torch
import torch.nn as nn

from models.stgcn.model.layers import (
    TemporalConv,
    GraphConv,
)


class STConvBlock(nn.Module):
    """
    STGCN Block

    Temporal
    -> Graph
    -> Temporal
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size=3,
        dropout=0.3,
    ):
        super().__init__()

        # -----------------------------
        # Temporal Conv 1
        # -----------------------------
        self.temp_conv1 = TemporalConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        # -----------------------------
        # Graph Conv
        # -----------------------------
        self.graph_conv = GraphConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
        )

        # -----------------------------
        # Temporal Conv 2
        # -----------------------------
        self.temp_conv2 = TemporalConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A_hat):

        # ---------------------------------
        # Temporal
        # ---------------------------------
        x = self.temp_conv1(x)

        # ---------------------------------
        # Graph
        # ---------------------------------
        x = self.graph_conv(x, A_hat)

        # ---------------------------------
        # Temporal
        # ---------------------------------
        x = self.temp_conv2(x)

        x = self.dropout(x)

        return x