"""Train a streaming velocity decoder on FALCON held-in sessions and score it on
the local minival split — a real R² with public labels, no submission required.

    python -m noema.eval.falcon_run --data data/000954 --task h1

Held-out (cross-session few-shot) scoring uses sequestered labels and needs an
EvalAI submission; this reports the local held-in number.
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


def _load_sessions(pattern, task):
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    sessions = []
    for f in sorted(glob.glob(pattern)):
        neural, kin, _, _ = load_nwb(f, FalconTask[task])
        sessions.append((neural.astype("float32"), kin.astype("float32")))
    if not sessions:
        raise FileNotFoundError(f"no sessions matched {pattern}")
    return sessions


def main():
    p = argparse.ArgumentParser(prog="noema.eval.falcon_run")
    p.add_argument("--data", default="data/000954")
    p.add_argument("--task", default="h1")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--enc-depth", type=int, default=4)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--w-behavior", type=float, default=5.0)
    p.add_argument("--multistep", type=int, default=0, help=">1 adds a multi-step world-model rollout loss")
    args = p.parse_args()

    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.evaluator import FalconEvaluator
    from falcon_challenge.interface import BCIDecoder

    cfg = FalconConfig(task=FalconTask[args.task])
    device = default_device()

    # Held-in calibration sessions, windowed. Velocity is tiny-scale (std ~1e-3), so
    # standardize per dim for a stable MSE, and undo it in the decoder to match the
    # evaluator's raw-velocity R².
    sessions = _load_sessions(f"{args.data}/*held-in-calib/*.nwb", args.task)
    all_kin = np.concatenate([k for _, k in sessions], 0)
    vmean, vstd = all_kin.mean(0), all_kin.std(0) + 1e-8
    parts = [SpikeWindows(n, behavior=(k - vmean) / vstd, window=args.window) for n, k in sessions]
    ds = ConcatDataset(parts)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=parts[0].collate, drop_last=True)

    model = Noema(dim=args.dim, enc_depth=args.enc_depth, wm_depth=2, heads=8,
                  max_units=cfg.n_channels, behavior_dim=cfg.out_dim, multistep=args.multistep).to(device)

    def log(step, d):
        if "loss_behavior" in d and step % 500 == 0:
            total = sum(v for k, v in d.items() if k.startswith("loss"))
            print(f"step {step:5d} loss_behavior={d['loss_behavior']:.4f} total={total:.3f}", flush=True)

    train(model, loader, TrainConfig(steps=args.steps, warmup=100, lr=args.lr, w_behavior=args.w_behavior,
                                     ckpt=""), device=device, on_log=log)

    vmean_t = torch.as_tensor(vmean, dtype=torch.float32, device=device)
    vstd_t = torch.as_tensor(vstd, dtype=torch.float32, device=device)

    class NoemaDecoder(BCIDecoder):
        def __init__(self):
            super().__init__(cfg, batch_size=1)
            self.stream = StreamingDecoder(model, torch.arange(cfg.n_channels), args.window, device)

        def reset(self, dataset_tags=[""]):
            self.stream.reset(len(dataset_tags))

        def predict(self, neural_observations):
            return (self.stream.step(neural_observations) * vstd_t + vmean_t).cpu().numpy()

        def on_done(self, dones):
            pass

    result = FalconEvaluator(eval_remote=False, split=args.task).evaluate(NoemaDecoder(), phase="minival")
    print(f"FALCON {args.task} minival result: {result}", flush=True)


if __name__ == "__main__":
    main()
