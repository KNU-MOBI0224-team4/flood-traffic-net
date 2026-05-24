import numpy as np
import pandas as pd
import torch


def load_adjacency_matrix(csv_path):
    """
    Load adjacency matrix from CSV.

    Returns:
        A_hat: normalized adjacency matrix
        A: original adjacency matrix
    """

    # -----------------------------
    # Load adjacency CSV
    # -----------------------------
    df = pd.read_csv(csv_path, index_col=0)

    A = df.values.astype(np.float32)

    print("Adjacency shape:", A.shape)

    # -----------------------------
    # Add self-loop
    # -----------------------------
    A = A + np.eye(A.shape[0], dtype=np.float32)

    # -----------------------------
    # Degree matrix
    # -----------------------------
    D = np.sum(A, axis=1)

    # -----------------------------
    # D^(-1/2)
    # -----------------------------
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))

    # -----------------------------
    # Symmetric normalization
    # -----------------------------
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    # -----------------------------
    # To torch tensor
    # -----------------------------
    A_hat = torch.from_numpy(A_hat).float()

    return A_hat