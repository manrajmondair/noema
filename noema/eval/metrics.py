"""Neural Latents Benchmark scoring, and the forward-prediction metrics."""

import math

import numpy as np



def bits_per_spike(rates, spikes):
    """Co-smoothing score: Poisson log-likelihood of held-out spikes under the
    predicted rates, over a per-neuron mean-rate null, in bits per spike.

    `rates` are expected counts per bin (lambda), not log-rates.
    """
    rates = rates.clamp_min(1e-8)
    null = spikes.mean(dim=tuple(range(spikes.dim() - 1)), keepdim=True).clamp_min(1e-8)
    ll = (spikes * rates.log() - rates).sum()
    ll_null = (spikes * null.log() - null).sum()
    return ((ll - ll_null) / spikes.sum().clamp_min(1) / math.log(2)).item()


def r2_score(pred, target):
    """Uniform-average R² across output dimensions (velocity decoding).

    Matches NLB, whose vel-R² is sklearn ``Ridge.score`` — i.e. ``r2_score`` with
    the default ``multioutput='uniform_average'``. Keep the unweighted mean.
    """
    pred = pred.reshape(-1, pred.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    res = ((target - pred) ** 2).sum(0)
    tot = ((target - target.mean(0)) ** 2).sum(0).clamp_min(1e-8)
    return (1 - res / tot).mean().item()


def r2_weighted(pred, target):
    """Variance-weighted R² — the FALCON convention.

    H1's seven kinematic dimensions differ in variance by an order of magnitude, so a
    uniform average is dragged down by the near-constant ones. Weighting by each
    dimension's share of total variance is what the challenge scorer rewards.
    """
    pred = pred.reshape(-1, pred.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    res = ((target - pred) ** 2).sum(0)
    tot = ((target - target.mean(0)) ** 2).sum(0)
    return (1 - res.sum() / tot.sum().clamp_min(1e-8)).item()


def weighted_neuron_corr(pred, true):
    """Variance-weighted per-neuron Pearson across the recording, at each horizon.

    The forward-prediction literature's headline form (Minnick et al. 2026, "Wt-r").
    For each neuron it correlates the forecast time series against the recorded one
    across all forecast points, then averages over neurons weighted by how much that
    neuron actually varies — so silent channels cannot dominate.

    This replaces a cross-neuron correlation taken within a single time bin, which we
    had been reporting. That form has a floor set by the recording rather than by the
    model: for Poisson counts with per-channel rates lambda, a flat forecast scores
    sqrt(Var(lambda) / (Var(lambda) + E[lambda])) — 0.02 on one dataset and 0.93 on
    another, with no model involved. It cannot be a headline.

    `pred` and `true` are [points, horizon, neurons].
    """
    out = np.full(pred.shape[1], np.nan)
    for h in range(pred.shape[1]):
        p, t = pred[:, h], true[:, h]
        weight = t.var(0)
        r = np.full(t.shape[1], np.nan)
        for c in range(t.shape[1]):
            if p[:, c].std() > 1e-9 and t[:, c].std() > 1e-9:
                r[c] = np.corrcoef(p[:, c], t[:, c])[0, 1]
        ok = np.isfinite(r) & (weight > 0)
        if ok.any():
            out[h] = float(np.average(r[ok], weights=weight[ok]))
    return out


def population_rate_corr(pred, true):
    """Correlation of the summed population rate over time, per horizon.

    Structurally immune to the static-profile floor for the same reason: summing over
    neurons first leaves one time series per horizon, so a per-neuron mean contributes
    a constant offset that correlation removes.
    """
    out = np.full(pred.shape[1], np.nan)
    for h in range(pred.shape[1]):
        p, t = pred[:, h].sum(1), true[:, h].sum(1)
        if p.std() > 1e-9 and t.std() > 1e-9:
            out[h] = float(np.corrcoef(p, t)[0, 1])
    return out
