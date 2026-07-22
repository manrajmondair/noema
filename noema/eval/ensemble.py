"""Rate-space ensembling.

Average the predicted firing rates (not log-rates) across models, then score
co-bps once. Poisson co-bps rewards accurate rates, and the mean rate is the
right pooling for the Poisson likelihood.
"""

import torch
from torch.utils.data import DataLoader

from .baselines import gaussian_smooth
from .metrics import bits_per_spike


@torch.no_grad()
def ensemble_rates(models, dataset, device=None, batch_size=64):
    """Mean predicted held-out firing rates across models, with targets."""
    device = device or next(models[0].parameters()).device
    for m in models:
        m.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate)

    rates, targets = [], []
    for batch in loader:
        counts, uid = batch["counts"].to(device), batch["unit_ids"].to(device)
        tgt = batch["target_unit_ids"].to(device)
        member = [m.cosmooth(counts, uid, tgt).exp() for m in models]
        rates.append(torch.stack(member).mean(0).cpu())
        targets.append(batch["target_counts"])
    return torch.cat(rates), torch.cat(targets)


@torch.no_grad()
def member_rates(models, dataset, device=None, batch_size=64, tta=0):
    """Per-member held-out rates [n_models] of [trials,T,N], with shared targets.
    tta>0 averages each member over `tta` coordinated-dropout masks (variance reduction, no retraining)."""
    device = device or next(models[0].parameters()).device
    for m in models:
        m.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate)
    per, targets = [[] for _ in models], []
    for batch in loader:
        counts, uid = batch["counts"].to(device), batch["unit_ids"].to(device)
        tgt = batch["target_unit_ids"].to(device)
        for i, m in enumerate(models):
            r = m.cosmooth_tta(counts, uid, tgt, tta) if tta else m.cosmooth(counts, uid, tgt).exp()
            per[i].append(r.cpu())
        targets.append(batch["target_counts"])
    return [torch.cat(p) for p in per], torch.cat(targets)


def greedy_ensemble(rates, targets, max_size=40):
    """Caruana greedy selection: pick members (with replacement) to maximize co-bps."""
    chosen, best = [], float("-inf")
    while len(chosen) < max_size:
        scores = [bits_per_spike(sum(rates[j] for j in chosen + [i]) / (len(chosen) + 1), targets)
                  for i in range(len(rates))]
        i = max(range(len(scores)), key=lambda k: scores[k])
        if scores[i] <= best + 1e-5:
            break
        best, _ = scores[i], chosen.append(i)
    return chosen or [max(range(len(rates)), key=lambda i: bits_per_spike(rates[i], targets))]


def ensemble_co_bps(models, dataset, device=None, batch_size=64, smooth=0.0):
    """Ensemble rates, optionally smoothed over time, scored as co-bps. Light
    temporal smoothing matches true (smooth) PSTHs and stacks with ensembling."""
    rates, targets = ensemble_rates(models, dataset, device, batch_size)
    if smooth > 0:
        rates = gaussian_smooth(rates, smooth)
    return bits_per_spike(rates, targets)
