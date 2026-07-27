"""World-model rollout fidelity on real FALCON data.

Noema is a *world model*: it predicts future neural population state, not just the
current behavior. This trains the model on FALCON H1 and then, from a seed window,
rolls the world model forward autoregressively (open loop) and measures how well the
imagined firing tracks the true future firing at increasing horizons — the property
that separates a forward model from a decoder.

    python -m noema.eval.falcon_worldmodel --data data/000954 --task h1
"""

import argparse
import glob

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .. import Noema
from ..data.dataset import SpikeWindows
from ..sim.rollout import imagine
from ..train import TrainConfig, train
from ..utils import default_device


def _load(pattern, task):
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    out = []
    for f in sorted(glob.glob(pattern)):
        neural, _, _, _ = load_nwb(f, FalconTask[task])
        out.append(neural.astype("float32"))
    if not out:
        raise FileNotFoundError(f"no sessions matched {pattern}")
    return out


@torch.no_grad()
def _rollout_fidelity(model, sessions, seed, horizon, device, stride=20):
    """Correlation between imagined and true firing at each horizon, over many seeds."""
    n_ch = sessions[0].shape[1]
    ids = torch.arange(n_ch, device=device)
    corr = np.zeros(horizon)
    counts = np.zeros(horizon)
    for neural in sessions:
        t = torch.as_tensor(neural, dtype=torch.float32, device=device)
        for s in range(0, len(t) - seed - horizon, stride):
            sd = t[s: s + seed].unsqueeze(0)
            fut = torch.zeros(1, horizon, 0, device=device)  # unconditioned autoregressive rollout
            rates, _ = imagine(model, sd, ids, fut)           # [1, horizon, n_ch]
            true = t[s + seed: s + seed + horizon]
            for h in range(horizon):
                a, b = rates[0, h].cpu().numpy(), true[h].cpu().numpy()
                if a.std() > 1e-6 and b.std() > 1e-6:
                    corr[h] += np.corrcoef(a, b)[0, 1]
                    counts[h] += 1
    return corr / np.maximum(counts, 1)


def main():
    p = argparse.ArgumentParser(prog="noema.eval.falcon_worldmodel")
    p.add_argument("--data", default="data/000954")
    p.add_argument("--task", default="h1")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--enc-depth", type=int, default=4)
    p.add_argument("--wm-depth", type=int, default=3)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--w-forecast", type=float, default=3.0, help="up-weight the observation-space forecast")
    p.add_argument("--multistep", type=int, default=0, help=">1 adds a multi-step rollout loss (drift resistance)")
    args = p.parse_args()

    from falcon_challenge.config import FalconConfig, FalconTask

    cfg = FalconConfig(task=FalconTask[args.task])
    device = default_device()

    train_sessions = _load(f"{args.data}/*held-in-calib/*.nwb", args.task)
    parts = [SpikeWindows(n, window=args.window) for n in train_sessions]
    loader = DataLoader(ConcatDataset(parts), batch_size=args.batch, shuffle=True,
                        collate_fn=parts[0].collate, drop_last=True)

    # world model matters here, so give it depth and weight the forecast/JEPA terms
    model = Noema(dim=args.dim, enc_depth=args.enc_depth, wm_depth=args.wm_depth, heads=8,
                  max_units=cfg.n_channels, multistep=args.multistep).to(device)

    def log(step, d):
        if "loss_forecast" in d and step % 1000 == 0:
            print(f"step {step:5d} forecast={d['loss_forecast']:.3f} jepa={d.get('loss_jepa', 0):.3f}", flush=True)

    train(model, loader, TrainConfig(steps=args.steps, warmup=100, lr=3e-4,
                                     w_forecast=args.w_forecast, ckpt=""), device=device, on_log=log)

    minival = _load(f"{args.data}/*held-in-minival/*.nwb", args.task)
    fid = _rollout_fidelity(model, minival, args.window, args.horizon, device)
    print("rollout firing-correlation vs horizon (bins):", flush=True)
    for h, c in enumerate(fid, 1):
        print(f"  h={h:2d}  corr={c:.3f}", flush=True)
    print(f"summary: h1={fid[0]:.3f}  h={len(fid)}={fid[-1]:.3f}  mean={fid.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
