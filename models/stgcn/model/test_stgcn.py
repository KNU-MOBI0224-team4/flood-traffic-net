import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../")
    )
)

import torch

from models.stgcn.model.stgcn import STGCN


B = 4
T = 12
N = 329
F = 8

x = torch.randn(B, T, N, F)

A_hat = torch.randn(N, N)

model = STGCN(
    num_nodes=N,
    in_channels=F,
)

out = model(x, A_hat)

print("Output shape:", out.shape)