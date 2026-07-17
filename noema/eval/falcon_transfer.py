"""Cross-session transfer study on FALCON H1 — the benchmark's 'real wall'.

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
def _score(model, neural, kin, mask, window, vmean, vstd, device):
    """Stream one bin at a time; variance-weighted R^2 on eval-mask timesteps."""
    from sklearn.metrics import r2_score

    stream = StreamingDecoder(model, torch.arange(neural.shape[1]), window, device)
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
    args = p.parse_args()

    from falcon_challenge.config import FalconConfig, FalconTask

    cfg = FalconConfig(task=FalconTask[args.task])
    device = default_device()

    sessions = _load(f"{args.data}/*held-in-calib/*.nwb", args.task)
    if args.held_out >= len(sessions):
        raise ValueError(f"only {len(sessions)} sessions; cannot hold out {args.held_out}")
    train_sessions, held_out = sessions[: -args.held_out], sessions[-args.held_out:]
    print(f"train on {len(train_sessions)} sessions, transfer to {len(held_out)} unseen: "
          f"{[s for s, *_ in held_out]}", flush=True)

    all_kin = np.concatenate([k for _, _, k, _ in train_sessions], 0)
    vmean, vstd = all_kin.mean(0), all_kin.std(0) + 1e-8
    parts = [SpikeWindows(n, behavior=(k - vmean) / vstd, window=args.window) for _, n, k, _ in train_sessions]
    loader = DataLoader(ConcatDataset(parts), batch_size=args.batch, shuffle=True,
                        collate_fn=parts[0].collate, drop_last=True)

    model = Noema(dim=args.dim, enc_depth=args.enc_depth, wm_depth=2, heads=8,
                  max_units=cfg.n_channels, behavior_dim=cfg.out_dim).to(device)

    def log(step, d):
        if "loss_behavior" in d and step % 1000 == 0:
            print(f"step {step:5d} loss_behavior={d['loss_behavior']:.4f}", flush=True)

    train(model, loader, TrainConfig(steps=args.steps, warmup=100, lr=3e-4,
                                     w_behavior=args.w_behavior, ckpt=""), device=device, on_log=log)

    # in-distribution check (a train session) vs zero-shot transfer (held-out sessions)
    seen = _score(model, *train_sessions[-1][1:], args.window, vmean, vstd, device)
    r2s = [_score(model, n, k, m, args.window, vmean, vstd, device) for _, n, k, m in held_out]
    print(f"seen-session R2 = {seen:.3f}", flush=True)
    print(f"zero-shot cross-session R2 = {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}  "
          f"per-session {[round(r, 3) for r in r2s]}", flush=True)


if __name__ == "__main__":
    main()
