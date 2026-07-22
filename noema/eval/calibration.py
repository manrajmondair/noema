"""Calibration-reduction curve.

Decode accuracy against the amount of calibration data, for a pretrained model
adapted few-shot versus a model trained from scratch. The gap at small budgets
is the practical payoff of pretraining for a foundation
model of neural activity.
"""

import copy

from torch.utils.data import DataLoader, Subset

from ..train import TrainConfig, few_shot_adapt, train
from .nlb import evaluate


def calibration_curve(pretrained, fresh_model, calib_ds, val_ds, budgets,
                      adapt_steps=150, scratch_steps=150, batch_size=8, device=None):
    curve = []
    for n in budgets:
        loader = DataLoader(Subset(calib_ds, range(n)), batch_size=min(n, batch_size),
                            shuffle=True, collate_fn=calib_ds.collate, drop_last=True)
        transfer = few_shot_adapt(copy.deepcopy(pretrained), loader, steps=adapt_steps, device=device)
        scratch = train(fresh_model(), list(loader),
                        TrainConfig(steps=scratch_steps, warmup=scratch_steps // 10, ckpt=""), device=device)
        curve.append({
            "trials": n,
            "transfer": evaluate(transfer, val_ds, device=device)["vel_r2"],
            "scratch": evaluate(scratch, val_ds, device=device)["vel_r2"],
        })
    return curve
