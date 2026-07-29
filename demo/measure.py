"""Forecast skill on a uniform grid of cuts, for the demo's calendar and curves.

One protocol for all thirteen recording days: same window, same horizon, same cut
spacing, same held-out status. The transfer study previously scored seen and unseen
sessions differently and put both on one axis, which made the size of the drift gap
partly an artefact of the protocol rather than a measurement of drift.

Skill is reported centred as well as raw. Raw correlation across channels is dominated
by the static per-channel firing profile, so a rollout that collapsed to a rescaled
population average would still score well; centring removes that profile and leaves only
credit for time-varying structure. Two baselines bracket the result: repeating the last
observed bin, and emitting the channel means the centring removes.
"""

import numpy as np
import torch

from noema.sim.rollout import imagine


def cut_grid(n_bins, window, horizon, count=100):
    """Evenly spaced cut positions with a full seed window behind and a full horizon ahead."""
    lo, hi = window, n_bins - horizon
    if hi <= lo:
        return np.empty(0, dtype=int)
    return np.unique(np.linspace(lo, hi, min(count, hi - lo), dtype=int))


def _corr(a, b):
    """Population correlation across channels within one bin."""
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def forecast_skill(model, neural, window, horizon, count=100, device=None):
    """Per-horizon skill of the model's open-loop rollout against what was recorded.

    Returns a dict of length-`horizon` arrays: centred, raw, persistence, channel_mean.
    """
    device = device or next(model.parameters()).device
    model.eval()
    tape = torch.as_tensor(np.asarray(neural), dtype=torch.float32, device=device)
    ids = torch.arange(tape.shape[1], device=device)
    cuts = cut_grid(len(tape), window, horizon, count)

    pred = [[] for _ in range(horizon)]
    true = [[] for _ in range(horizon)]
    last = [[] for _ in range(horizon)]
    for cut in cuts:
        seed = tape[cut - window:cut].unsqueeze(0)
        rates, _ = imagine(model, seed, ids, torch.zeros(1, horizon, 0, device=device))
        for h in range(horizon):
            pred[h].append(rates[0, h].cpu().numpy())
            true[h].append(tape[cut + h].cpu().numpy())
            last[h].append(tape[cut - 1].cpu().numpy())  # persistence: repeat the last bin seen

    out = {k: np.full(horizon, np.nan) for k in ("centred", "raw", "persistence", "channel_mean")}
    for h in range(horizon):
        p, t, prev = np.stack(pred[h]), np.stack(true[h]), np.stack(last[h])
        profile = t.mean(0)  # the static per-channel firing profile, across cuts
        out["raw"][h] = np.nanmean([_corr(a, b) for a, b in zip(p, t)])
        pc, tc = p - p.mean(0), t - t.mean(0)
        out["centred"][h] = np.nanmean([_corr(a, b) for a, b in zip(pc, tc)])
        out["persistence"][h] = np.nanmean([_corr(a, b) for a, b in zip(prev - prev.mean(0), tc)])
        # The channel-mean baseline is constant across cuts, so it has no centred
        # variation at all. Reporting it raw is the point: it is the score a model
        # gets for learning nothing but the average firing rate of each channel.
        out["channel_mean"][h] = np.nanmean([_corr(profile, b) for b in t])
    return out


def calendar(model, tapes, window, horizon, count=100, device=None):
    """Skill per recording day. `tapes` is [(day_offset, split, neural), ...]."""
    return [{"day": day, "split": split,
             **{k: v.tolist() for k, v in forecast_skill(model, neural, window, horizon,
                                                         count, device).items()}}
            for day, split, neural in tapes]
