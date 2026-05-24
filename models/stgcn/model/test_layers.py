import torch
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../")
    )
)
from models.stgcn.model.layer import TemporalConv, GraphConv


B = 4
T = 12
N = 329
F = 8

x = torch.randn(B, T, N, F)

A_hat = torch.randn(N, N)

# -----------------------------
# Temporal Conv
# -----------------------------
temp = TemporalConv(
    in_channels=8,
    out_channels=32,
)

x = temp(x)

print("After TemporalConv:", x.shape)

# -----------------------------
# Graph Conv
# -----------------------------
gconv = GraphConv(
    in_channels=32,
    out_channels=64,
)

x = gconv(x, A_hat)

print("After GraphConv:", x.shape)