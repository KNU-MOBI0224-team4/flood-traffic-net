import numpy as np

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    recall_score,
)


def compute_metrics(
    logits,
    targets,
    mask,
    threshold=0.5,
):

    # ---------------------------------
    # sigmoid
    # ---------------------------------
    probs = 1 / (1 + np.exp(-logits))

    # ---------------------------------
    # masking
    # ---------------------------------
    probs = probs[mask == 1]
    targets = targets[mask == 1]

    # ---------------------------------
    # binary prediction
    # ---------------------------------
    preds = (probs >= threshold).astype(int)

    metrics = {}

    # ---------------------------------
    # PR-AUC
    # ---------------------------------
    metrics["pr_auc"] = average_precision_score(
        targets,
        probs,
    )

    # ---------------------------------
    # ROC-AUC
    # ---------------------------------
    metrics["roc_auc"] = roc_auc_score(
        targets,
        probs,
    )

    # ---------------------------------
    # F1
    # ---------------------------------
    metrics["f1"] = f1_score(
        targets,
        preds,
        zero_division=0,
    )

    # ---------------------------------
    # Recall
    # ---------------------------------
    metrics["recall"] = recall_score(
        targets,
        preds,
        zero_division=0,
    )

    return metrics