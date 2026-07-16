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
def ensemble_co_bps(models, dataset, device=None, batch_size=64, smooth=0.0):
    """Mean of member rates, optionally smoothed over time, scored as co-bps.
    Light temporal smoothing matches true (smooth) PSTHs and stacks with ensembling."""
    device = device or next(models[0].parameters()).device
    for m in models:
        m.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate)

    rates, targets = [], []
    for batch in loader:
        counts, uid = batch["counts"].to(device), batch["unit_ids"].to(device)
        tgt = batch["target_unit_ids"].to(device)
        member = [m.tokenizer.decode(m.encode(counts, uid), tgt).exp() for m in models]
        rates.append(torch.stack(member).mean(0).cpu())
        targets.append(batch["target_counts"])

    rates = gaussian_smooth(torch.cat(rates), smooth) if smooth > 0 else torch.cat(rates)
    return bits_per_spike(rates, torch.cat(targets))
