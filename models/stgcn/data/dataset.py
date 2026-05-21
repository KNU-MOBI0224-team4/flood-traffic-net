import numpy as np
import torch
from torch.utils.data import Dataset


class FloodSTGCNDataset(Dataset):
    """
    STGCN Dataset for Flood-induced Road Paralysis Onset Prediction

    Input:
        X[t-11:t]  -> shape [12, N, F]

    Target:
        z[t+1]     -> shape [N]

    Mask:
        z_mask[t+1] -> shape [N]
    """

    def __init__(
        self,
        x_path,
        target_path,
        seq_len=12,
        pred_horizon=1,
        split="train",
    ):

        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

        # -----------------------------
        # Load dynamic feature tensor
        # -----------------------------
        x_data = np.load(x_path)

        # expected:
        # X_dynamic shape = [T, N, F]
        self.X = x_data["X"]

        # -----------------------------
        # Load targets
        # -----------------------------
        target_data = np.load(target_path)

        self.z = target_data["z"]
        self.z_mask = target_data["z_mask"]

        # optional split array
        self.split = target_data["split"]

        # -----------------------------
        # Select valid indices
        # -----------------------------
        self.valid_indices = []

        T = self.X.shape[0]

        for t in range(seq_len, T - pred_horizon):

            target_t = t + pred_horizon

            # split filtering
            if self.split[target_t] != split:
                continue

            # at least one valid node
            if self.z_mask[target_t].sum() == 0:
                continue

            self.valid_indices.append(t)

        print(f"[{split}] valid samples: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):

        t = self.valid_indices[idx]

        x_start = t - self.seq_len
        x_end = t

        target_t = t + self.pred_horizon

        # -----------------------------
        # Input sequence
        # shape: [12, N, F]
        # -----------------------------
        x = self.X[x_start:x_end]

        # -----------------------------
        # Target
        # shape: [N]
        # -----------------------------
        y = self.z[target_t]

        # -----------------------------
        # Mask
        # shape: [N]
        # -----------------------------
        mask = self.z_mask[target_t]

        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y),
            torch.BoolTensor(mask),
        )