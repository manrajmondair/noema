"""Evaluate a trained model with the Neural Latents metrics."""

import torch
from torch.utils.data import DataLoader

from .metrics import bits_per_spike, r2_score


@torch.no_grad()
def evaluate(model, dataset, device=None, batch_size=64):
    device = device or next(model.parameters()).device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate)

    rates, targets, vel_pred, vel_true = [], [], [], []
    for batch in loader:
        z = model.encode(batch["counts"].to(device), batch["unit_ids"].to(device))
        if "target_counts" in batch:
            rate = model.tokenizer.decode(z, batch["target_unit_ids"].to(device)).exp()
            rates.append(rate.cpu())
            targets.append(batch["target_counts"])
        if model.behavior is not None and "behavior" in batch:
            vel_pred.append(model.behavior(z).cpu())
            vel_true.append(batch["behavior"])

    metrics = {}
    if rates:
        metrics["co_bps"] = bits_per_spike(torch.cat(rates), torch.cat(targets))
    if vel_pred:
        metrics["vel_r2"] = r2_score(torch.cat(vel_pred), torch.cat(vel_true))
    return metrics
