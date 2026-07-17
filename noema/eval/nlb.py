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
        counts, uid = batch["counts"].to(device), batch["unit_ids"].to(device)
        z = model.encode(counts, uid)
        if "target_counts" in batch:
            rate = model.cosmooth(counts, uid, batch["target_unit_ids"].to(device)).exp()
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


@torch.no_grad()
def _infer_rates(model, dataset, device, batch_size=64):
    """Model's inferred firing rates for the full population, with behavior."""
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate)
    rates, behavior = [], []
    for batch in loader:
        counts, in_ids = batch["counts"].to(device), batch["unit_ids"].to(device)
        # each readout via the model's own path (per-unit / cross for the spatial encoder)
        tokens, z = model._represent(counts, in_ids)
        hi = (model.tokenizer.decode_units(tokens, in_ids) if model.spatial else model.tokenizer.decode(z, in_ids))
        parts = [hi.exp()]
        if "target_unit_ids" in batch:
            parts.append(model._cosmooth_from(tokens, z, batch["target_unit_ids"].to(device)).exp())
        rates.append(torch.cat(parts, dim=-1).reshape(-1, sum(p.size(-1) for p in parts)).cpu())
        behavior.append(batch["behavior"].reshape(-1, batch["behavior"].size(-1)))
    return torch.cat(rates).numpy(), torch.cat(behavior).numpy()


def official_velocity_r2(model, train_ds, val_ds, device=None):
    """Leaderboard velocity R²: the official NLB ridge decoder (GridSearchCV over
    alpha) fit from inferred rates to hand velocity, not our learned head."""
    from nlb_tools.evaluation import fit_and_eval_decoder
    device = device or next(model.parameters()).device
    model.eval()
    tr_rates, tr_vel = _infer_rates(model, train_ds, device)
    ev_rates, ev_vel = _infer_rates(model, val_ds, device)
    # Score against raw kinematics (what the leaderboard uses), not the per-split
    # standardized velocity — R² is only affine-invariant under a shared target frame.
    tr_vel, ev_vel = _raw_behavior(tr_vel, train_ds), _raw_behavior(ev_vel, val_ds)
    return float(fit_and_eval_decoder(tr_rates, tr_vel, ev_rates, ev_vel))


def _raw_behavior(vel, ds):
    if getattr(ds, "behavior_stats", None) is None:
        return vel
    mean, std = (s.numpy() for s in ds.behavior_stats)
    return vel * std + mean
