"""Standard decoding baselines.

A ridge regression from (Gaussian-smoothed) spike counts to velocity is the
classic BCI reference point; a learned model has to beat it to be worth anything.
"""

import torch
import torch.nn.functional as F

from .metrics import r2_score


def gaussian_smooth(x, sigma):
    if sigma <= 0:
        return x
    k = int(sigma * 4) | 1  # odd kernel
    t = torch.arange(k, dtype=torch.float32) - k // 2
    w = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    w = (w / w.sum()).view(1, 1, -1)
    trials, steps, units = x.shape
    xp = x.transpose(1, 2).reshape(trials * units, 1, steps)
    xp = F.conv1d(xp, w, padding=k // 2)
    return xp.reshape(trials, units, steps).transpose(1, 2)


def ridge_velocity(train_ds, val_ds, alpha=1.0, sigma=2.0):
    """Fit ridge on the train split, return velocity R² on the val split."""
    def design(ds):
        x = gaussian_smooth(ds.heldin, sigma).reshape(-1, ds.heldin.size(-1))
        x = torch.cat([x, torch.ones(x.size(0), 1)], dim=1)  # bias column
        return x, ds.behavior.reshape(-1, ds.behavior.size(-1))

    xtr, ytr = design(train_ds)
    xva, yva = design(val_ds)
    gram = xtr.t() @ xtr + alpha * torch.eye(xtr.size(1))
    weights = torch.linalg.solve(gram, xtr.t() @ ytr)
    return r2_score(xva @ weights, yva)
