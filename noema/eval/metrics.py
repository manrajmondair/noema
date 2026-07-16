"""Neural Latents Benchmark scoring."""

import math



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
