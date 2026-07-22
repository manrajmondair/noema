"""Cross-session transfer study on FALCON H1.

Train the streaming velocity decoder on a subset of the held-in sessions and score
it, zero-shot, on the remaining (unseen) sessions. Same subject, different recording
days, so this measures robustness to the electrode/signal drift that the official
held-out split tests. Velocity R^2 is variance-weighted and restricted to the
evaluation mask, matching the FALCON scorer.

    python -m noema.eval.falcon_transfer --data data/000954 --task h1 --held-out 3
"""

import argparse
import glob

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .. import Noema
from ..data.dataset import SpikeWindows
from ..train import TrainConfig, train
from ..utils import default_device
from .streaming import StreamingDecoder


def _load(pattern, task):
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    out = []
    for f in sorted(glob.glob(pattern)):
        neural, kin, _, mask = load_nwb(f, FalconTask[task])
        out.append((f.split("/")[-1][-28:-4], neural.astype("float32"), kin.astype("float32"), mask))
    if not out:
        raise FileNotFoundError(f"no sessions matched {pattern}")
    return out


@torch.no_grad()
def _latents(model, neural, window, device):
    """The encoder's last-position latent over a session's windows -> [n_windows, dim]."""
    ds = SpikeWindows(neural, window=window)
    zs = []
    for b in DataLoader(ds, batch_size=64, collate_fn=ds.collate):
        z = model.encode(b["counts"].to(device), b["unit_ids"].to(device))
        zs.append(z[:, -1].cpu())
    return torch.cat(zs)


def _latent_stats(model, neural, window, device):
    z = _latents(model, neural, window, device)
    return z.mean(0), z.std(0) + 1e-4


def _coral_transform(train_z, unseen_z, device, eps=1e-3, shrink=1.0):
    """CORAL: whiten the unseen latents and recolor to the training covariance so the
    full second-order statistics match — closed-form, no adversarial instability. A full
    transform breaks the fixed decoder, so `shrink` blends the alignment toward identity
    (shrink=0 -> no change, 1 -> full CORAL), keeping the map near the decodable manifold."""
    dim = train_z.shape[1]
    tm, um = train_z.mean(0), unseen_z.mean(0)

    def msqrt(z, inv):
        c = torch.cov(z.T) + eps * torch.eye(dim)
        val, vec = torch.linalg.eigh(c)
        val = val.clamp_min(eps)
        d = val.rsqrt() if inv else val.sqrt()
        return (vec * d) @ vec.T

    a = msqrt(unseen_z, inv=True) @ msqrt(train_z, inv=False)
    a = ((1 - shrink) * torch.eye(dim) + shrink * a).to(device)
    tm, um = tm.to(device), um.to(device)
    return lambda zl: (zl - um) @ a + tm


@torch.no_grad()
def _score(model, neural, kin, mask, window, vmean, vstd, device, latent_transform=None):
    """Stream one bin at a time; variance-weighted R^2 on eval-mask timesteps."""
    from sklearn.metrics import r2_score

    stream = StreamingDecoder(model, torch.arange(neural.shape[1]), window, device,
                              latent_transform=latent_transform)
    stream.reset(1)
    pred = np.empty_like(kin)
    for t in range(neural.shape[0]):
        pred[t] = stream.step(neural[t:t + 1])[0].cpu().numpy() * vstd + vmean
    m = mask.astype(bool) if mask is not None else np.ones(len(kin), bool)
    return float(r2_score(kin[m], pred[m], multioutput="variance_weighted"))


def main():
    p = argparse.ArgumentParser(prog="noema.eval.falcon_transfer")
    p.add_argument("--data", default="data/000954")
    p.add_argument("--task", default="h1")
    p.add_argument("--held-out", type=int, default=3, help="trailing sessions to hold out from training")
    p.add_argument("--window", type=int, default=75)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--enc-depth", type=int, default=5)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--w-behavior", type=float, default=6.0)
    p.add_argument("--calib-frac", type=float, default=0.5, help="fraction of each held-out session used for calibration")
    p.add_argument("--adapt-steps", type=int, default=0, help=">0 enables few-shot adaptation on each session's calib split")
    p.add_argument("--adapt-lr", type=float, default=3e-4)
    p.add_argument("--adversary", action="store_true", help="domain-adversarial session invariance during training")
    p.add_argument("--adv-weight", type=float, default=1.0)
    p.add_argument("--input-norm", action="store_true",
                   help="per-session per-channel gain normalization (equalizes electrode drift, log1p-safe)")
    p.add_argument("--latent-align", action="store_true",
                   help="NoMAD-style: match each unseen session's latent moments to the training distribution")
    p.add_argument("--coral-align", action="store_true",
                   help="CORAL: match full latent covariance (whiten-recolor) of each unseen session to training")
    p.add_argument("--coral-shrink", type=float, default=1.0, help="blend CORAL toward identity (0=off, 1=full)")
    args = p.parse_args()

    from falcon_challenge.config import FalconConfig, FalconTask

    cfg = FalconConfig(task=FalconTask[args.task])
    device = default_device()

    sessions = _load(f"{args.data}/*held-in-calib/*.nwb", args.task)
    if args.held_out >= len(sessions):
        raise ValueError(f"only {len(sessions)} sessions; cannot hold out {args.held_out}")
    train_sessions, held_out = sessions[: -args.held_out], sessions[-args.held_out:]
    if args.input_norm:
        # Divide each channel by its mean firing: equalizes per-channel gain drift, keeps
        # counts non-negative for log1p. For unseen sessions the normalizer is estimated
        # from the calibration portion ONLY (causal, no eval-split statistics leak).
        def _norm(n, ref):
            return n / (ref.mean(0, keepdims=True) + 1e-3)
        train_sessions = [(nm, _norm(n, n), k, m) for nm, n, k, m in train_sessions]
        held_out = [(nm, _norm(n, n[: int(len(n) * args.calib_frac)]), k, m) for nm, n, k, m in held_out]
    print(f"train on {len(train_sessions)} sessions, transfer to {len(held_out)} unseen: "
          f"{[s for s, *_ in held_out]}", flush=True)

    all_kin = np.concatenate([k for _, _, k, _ in train_sessions], 0)
    vmean, vstd = all_kin.mean(0), all_kin.std(0) + 1e-8
    parts = [SpikeWindows(n, behavior=(k - vmean) / vstd, window=args.window,
                          session=(i if args.adversary else None))
             for i, (_, n, k, _) in enumerate(train_sessions)]
    loader = DataLoader(ConcatDataset(parts), batch_size=args.batch, shuffle=True,
                        collate_fn=parts[0].collate, drop_last=True)

    model = Noema(dim=args.dim, enc_depth=args.enc_depth, wm_depth=2, heads=8,
                  max_units=cfg.n_channels, behavior_dim=cfg.out_dim,
                  sessions=(len(train_sessions) if args.adversary else 0),
                  adv_weight=args.adv_weight).to(device)

    def log(step, d):
        if "loss_behavior" in d and step % 1000 == 0:
            print(f"step {step:5d} loss_behavior={d['loss_behavior']:.4f}", flush=True)

    train(model, loader, TrainConfig(steps=args.steps, warmup=100, lr=3e-4,
                                     w_behavior=args.w_behavior, ckpt=""), device=device, on_log=log)

    import copy

    train_lm = train_ls = train_z = None
    if args.latent_align or args.coral_align:  # reference latent distribution from training sessions
        ref = np.concatenate([n for _, n, _, _ in train_sessions[:4]], 0)
        if args.coral_align:
            train_z = _latents(model, ref, args.window, device)
        else:
            train_lm, train_ls = (x.to(device) for x in _latent_stats(model, ref, args.window, device))

    seen = _score(model, *train_sessions[-1][1:], args.window, vmean, vstd, device)
    print(f"seen-session R2 = {seen:.3f}", flush=True)

    # Score each unseen session on its held-out (eval) portion. Zero-shot uses the
    # base model; latent-align maps its latents to the training moments; few-shot
    # additionally adapts a copy on the session's calibration portion.
    zero, few = [], []
    for _, n, k, m in held_out:
        cut = int(len(n) * args.calib_frac)
        me = None if m is None else m[cut:]
        transform = None
        if args.coral_align and cut > args.window * 4:
            transform = _coral_transform(train_z, _latents(model, n[:cut], args.window, device),
                                         device, shrink=args.coral_shrink)
        elif args.latent_align and cut > args.window * 4:
            um, us = (x.to(device) for x in _latent_stats(model, n[:cut], args.window, device))
            transform = (lambda zl, a=um, b=us: (zl - a) / b * train_ls + train_lm)
        zero.append(_score(model, n[cut:], k[cut:], me, args.window, vmean, vstd, device, transform))
        if args.adapt_steps > 0 and cut > args.window * 4:
            adapted = copy.deepcopy(model)
            cds = SpikeWindows(n[:cut], behavior=(k[:cut] - vmean) / vstd, window=args.window)
            cl = DataLoader(cds, batch_size=args.batch, shuffle=True, collate_fn=cds.collate, drop_last=True)
            train(adapted, cl, TrainConfig(steps=args.adapt_steps, warmup=10, lr=args.adapt_lr,
                                           w_behavior=args.w_behavior, ckpt=""), device=device)
            few.append(_score(adapted, n[cut:], k[cut:], me, args.window, vmean, vstd, device, transform))
    print(f"zero-shot cross-session R2 = {np.mean(zero):.3f} +/- {np.std(zero):.3f}  {[round(r, 3) for r in zero]}", flush=True)
    if few:
        print(f"few-shot ({args.adapt_steps}-step) cross-session R2 = {np.mean(few):.3f} +/- {np.std(few):.3f}  "
              f"{[round(r, 3) for r in few]}", flush=True)


if __name__ == "__main__":
    main()
