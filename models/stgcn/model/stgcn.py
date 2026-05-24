import torch
import torch.nn as nn

from models.stgcn.model.layers import TemporalConv
from models.stgcn.model.stgcn_block import STConvBlock


class STGCN(nn.Module):

    def __init__(
        self,
        num_nodes,
        in_channels,
        hidden_channels=32,
        out_channels=64,
        kernel_size=3,
        dropout=0.3,
    ):
        super().__init__()

        # ---------------------------------
        # ST Block 1
        # ---------------------------------
        self.block1 = STConvBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # ---------------------------------
        # ST Block 2
        # ---------------------------------
        self.block2 = STConvBlock(
            in_channels=hidden_channels,
            hidden_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # ---------------------------------
        # Final Temporal Conv
        # ---------------------------------
        self.final_temp = TemporalConv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

        # ---------------------------------
        # Final Prediction Head
        # ---------------------------------
        self.fc = nn.Linear(out_channels, 1)

    def forward(self, x, A_hat):

        # ---------------------------------
        # ST Block 1
        # ---------------------------------
        x = self.block1(x, A_hat)

        # ---------------------------------
        # ST Block 2
        # ---------------------------------
        x = self.block2(x, A_hat)

        # ---------------------------------
        # Final Temporal
        # ---------------------------------
        x = self.final_temp(x)

        # ---------------------------------
        # Use last timestep only
        # shape:
        # [B, N, C]
        # ---------------------------------
        x = x[:, -1]

        # ---------------------------------
        # Node-wise prediction
        # [B, N, 1]
        # ---------------------------------
        x = self.fc(x)

        # ---------------------------------
        # [B, N]
        # ---------------------------------
        x = x.squeeze(-1)

        return x