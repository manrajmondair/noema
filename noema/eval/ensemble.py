"""Rate-space ensembling.

Average the predicted firing rates (not log-rates) across models, then score
co-bps once. Poisson co-bps rewards accurate rates, and the mean rate is the
right pooling — every top NLB entry above ~0.36 is a rate ensemble.
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


def ensemble_co_bps(models, dataset, device=None, batch_size=64, smooth=0.0):
    """Ensemble rates, optionally smoothed over time, scored as co-bps. Light
    temporal smoothing matches true (smooth) PSTHs and stacks with ensembling."""
    rates, targets = ensemble_rates(models, dataset, device, batch_size)
    if smooth > 0:
        rates = gaussian_smooth(rates, smooth)
    return bits_per_spike(rates, targets)
