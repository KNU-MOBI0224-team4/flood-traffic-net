import torch
import numpy as np

from tqdm import tqdm

from models.stgcn.utils.metrics import compute_metrics


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    A_hat,
    device,
):

    model.train()

    total_loss = 0

    for x, y, mask in tqdm(loader):

        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        # ---------------------------------
        # forward
        # ---------------------------------
        logits = model(x, A_hat)

        # ---------------------------------
        # loss
        # ---------------------------------
        if torch.isnan(logits).any():
            print("🚨 NaN in logits")
            print("logits min:", logits.min().item())
            print("logits max:", logits.max().item())
            exit()

        if torch.isinf(logits).any():
            print("🚨 Inf in logits")
            exit()

        loss = criterion(
            logits,
            y,
            mask,
        )

        if torch.isnan(loss):
            print("🚨 NaN in loss")

            print("y min:", y.min().item())
            print("y max:", y.max().item())

            print("mask sum:", mask.sum().item())

            exit()

        # ---------------------------------
        # backward
        # ---------------------------------
        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5.0
        )
        
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    A_hat,
    device,
):

    model.eval()

    total_loss = 0

    all_logits = []
    all_targets = []
    all_masks = []

    for x, y, mask in tqdm(loader):

        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        logits = model(x, A_hat)

        loss = criterion(
            logits,
            y,
            mask,
        )

        total_loss += loss.item()

        all_logits.append(
            logits.cpu().numpy()
        )

        all_targets.append(
            y.cpu().numpy()
        )

        all_masks.append(
            mask.cpu().numpy()
        )

    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    masks = np.concatenate(all_masks)

    metrics = compute_metrics(
        logits,
        targets,
        masks,
    )

    metrics["loss"] = total_loss / len(loader)

    return metrics