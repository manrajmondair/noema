"""World-model rollout fidelity and decoder-in-imagination on real FALCON data.

Noema is a *world model*: it predicts future neural population state, not just the
current behavior. This trains the model on FALCON H1 and then, from a seed window,
rolls the world model forward autoregressively (open loop) to measure two things:

  --fidelity  how well imagined firing tracks true firing at increasing horizons
  --sim2real  whether a decoder trained ONLY on imagined data decodes real recordings

    python -m noema.eval.falcon_worldmodel --data data/000954 --task h1 --sim2real
"""

import argparse

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .. import Noema
from ..data.dataset import SpikeWindows
from ..sim.rollout import imagine
from ..train import TrainConfig, train
from ..utils import default_device
from .baselines import ridge_velocity
from .falcon import disjoint_calib, load_sessions
from .metrics import r2_weighted
from .sim2real import imagine_windows


def _corr(a, b):
    return np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else np.nan


@torch.no_grad()
def _rollout_fidelity(model, sessions, seed, horizon, device, stride=20):
    """Imagined-vs-true population-pattern correlation per horizon, against two floors.

    Each score correlates a predicted population vector with the true one WITHIN a time
    bin, averaged over windows. Raw correlation is dominated by the static per-channel
    mean rate, so a rollout collapsing toward a rescaled population average still scores
    well — a channel-mean baseline reaches about 0.59.

    Centring removes a per-channel profile, but WHICH profile decides what the number
    means. Pooling every session and removing one grand profile leaves each session's
    own level intact, and that difference alone then reads as predicted structure: a
    constant per-session profile scored that way reaches 0.24, matching what the model
    was credited with. So the profile is removed per session, and the two constant
    predictors that ought to score nothing are reported beside the model rather than
    left to be assumed away.
    """
    ids = torch.arange(sessions[0][1].shape[1], device=device)
    arms = ("model", "seed_mean", "persistence")
    got = {a: [[] for _ in range(horizon)] for a in arms}
    truth = [[] for _ in range(horizon)]
    dropped = 0

    for _, neural, _, _ in sessions:
        t = torch.as_tensor(neural, dtype=torch.float32, device=device)
        # Per session: the profile removed below, and where each window's rows land.
        profile = t.mean(0).cpu().numpy()
        for s in range(0, len(t) - seed - horizon, stride):
            window = t[s:s + seed]
            rates, _ = imagine(model, window.unsqueeze(0), ids,
                               torch.zeros(1, horizon, 0, device=device))
            # Two predictors with no time-varying content, both strictly causal: the
            # mean of the seed window, and its last bin repeated.
            flat = window.mean(0).cpu().numpy()
            last = window[-1].cpu().numpy()
            for h in range(horizon):
                got["model"][h].append(rates[0, h].cpu().numpy() - profile)
                got["seed_mean"][h].append(flat - profile)
                got["persistence"][h].append(last - profile)
                truth[h].append(t[s + seed + h].cpu().numpy() - profile)

    out = {a: np.zeros(horizon) for a in arms}
    for h in range(horizon):
        q = np.stack(truth[h])
        for a in arms:
            scores = [_corr(x, y) for x, y in zip(np.stack(got[a][h]), q)]
            dropped += int(np.isnan(scores).sum()) if a == "model" else 0
            out[a][h] = np.nanmean(scores)
    out["dropped"] = dropped
    return out


def _windows(neural, kin, mask, window, horizon, stride):
    """Seed windows plus the real spikes and kinematics of the `horizon` bins after them,
    so the imagined and real arms are aligned bin for bin."""
    seeds, counts, kins = [], [], []
    for s in range(0, len(neural) - window - horizon, stride):
        e = s + window
        if mask is not None and not mask[e:e + horizon].all():
            continue  # only score bins the challenge scorer would score
        seeds.append(neural[s:e]), counts.append(neural[e:e + horizon]), kins.append(kin[e:e + horizon])
    stack = lambda a: torch.as_tensor(np.stack(a), dtype=torch.float32)
    return stack(seeds), stack(counts), stack(kins)


def _factorial(model, ids, seeds, real_counts, real_kin, val_ds, horizon, samples, stats, device):
    """Cross spikes {real, imagined} with labels {real, model's head}, plus a floor.

    A single sim-to-real number cannot say *which* half of the simulator failed. The
    gap R-D isolates the cost of generating spikes, R-S the cost of generating labels.
    S is also the control that matters most: if I is no better than S, the world model
    added nothing over pseudo-labelling and there is no simulator claim to make.
    """
    mean, std = stats
    # Roll out ONCE. The imagined spikes do not depend on which labels they are paired
    # with, so every imagined arm shares one Poisson draw — that makes D-vs-I an exact
    # contrast in the labels rather than one blurred by independent sampling noise.
    imagined = imagine_windows(model, ids, seeds, horizon, samples=samples, device=device)
    n = seeds.size(0)
    # The head emits standardized kinematics; scoring happens in raw units, so undo it.
    head_kin = imagined.behavior[:n] * std + mean
    rep = lambda x: x.repeat(samples, 1, 1)
    sim = lambda labels: SpikeWindows(imagined.heldin, behavior=rep(labels))
    real = lambda labels: SpikeWindows(real_counts, behavior=labels)
    fit = lambda ds: ridge_velocity(ds, val_ds, score=r2_weighted)
    arms = {
        "R real spikes / real labels ": fit(real(real_kin)),
        "S real spikes / head labels ": fit(real(head_kin)),
        "D imagined   / real labels ": fit(sim(real_kin)),
        "I imagined   / head labels ": fit(sim(head_kin)),
        "N imagined   / shuffled    ": fit(sim(head_kin[torch.randperm(n)])),
    }
    ratio = lambda a, b: (a.mean() / b.mean().clamp_min(1e-8)).item()
    # Read the head off ENCODER latents for the same bins. The head is only ever trained
    # on encoder latents but is read off world-model latents during rollout, so comparing
    # the two separates "the head never learned" (both collapse) from "the head is being
    # evaluated off-manifold" (in-distribution is healthy, rollout is not).
    with torch.no_grad():
        z = model.encode(real_counts.to(device), ids.to(device))
        indist_kin = model.behavior(z).cpu() * std + mean
    diagnostics = {
        # near 0 means the labels collapsed to a constant, which makes I and N coincide
        "label_std_ratio": ratio(head_kin.std(0), real_kin.std(0)),
        "head_std_indist": ratio(indist_kin.std(0), real_kin.std(0)),
        "rate_mean_ratio": ratio(imagined.heldin.mean(0), real_counts.mean(0)),
        "rate_chan_std_ratio": ratio(imagined.heldin.std(1), real_counts.std(1)),
    }
    return arms, diagnostics


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
    p.add_argument("--w-behavior", type=float, default=5.0)
    p.add_argument("--multistep", type=int, default=0, help=">1 adds a multi-step rollout loss (drift resistance)")
    p.add_argument("--sim2real", action="store_true", help="run the decoder-in-imagination factorial")
    p.add_argument("--fidelity", action="store_true", help="run the rollout firing-correlation sweep")
    p.add_argument("--horizons", default="5,10,20", help="sim2real horizons to sweep")
    p.add_argument("--stride", type=int, default=10, help="bins between seed windows")
    p.add_argument("--samples", type=int, default=1, help="Poisson draws per imagined rate path")
    p.add_argument("--sessions", type=int, default=0, help="limit sessions (0 = all; use 1 for a smoke test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default="", help="save the trained weights here, so the model that "
                                              "was measured is the model that ships")
    p.add_argument("--train-files", default="", help="comma-separated NWB paths to train on, "
                                                     "instead of every calibration session")
    args = p.parse_args()

    from falcon_challenge.config import FalconConfig, FalconTask

    torch.manual_seed(args.seed)
    cfg = FalconConfig(task=FalconTask[args.task])
    device = default_device()

    minival = load_sessions(f"{args.data}/*held-in-minival/*.nwb", args.task)
    # minival is a byte-identical prefix of calib here; without this the rollout and
    # sim2real arms are all scored on bins the model trained on.
    if args.train_files:
        # An explicit file list, so a caller that also ships recordings can hold some
        # back. Naming the training set in one place is what keeps a held-out recording
        # from quietly being one the model read.
        #
        # The excision applies here too. This branch used to skip it while still scoring
        # on minival, and the recordings the demo trained on were the full calib files —
        # so roughly 45% of the evaluation windows came from bins the model had read.
        named = [s for path in args.train_files.split(",") for s in load_sessions(path, args.task)]
        overlapping = [m for m in minival if m[0] in {n[0] for n in named}]
        calib = disjoint_calib(named, overlapping) if overlapping else named
    else:
        calib = disjoint_calib(load_sessions(f"{args.data}/*held-in-calib/*.nwb", args.task), minival)
    if args.sessions:
        calib, minival = calib[:args.sessions], minival[:args.sessions]

    # Standardize kinematics on calibration statistics only — never the evaluation split.
    all_kin = np.concatenate([k for _, _, k, _ in calib], 0)
    vmean, vstd = all_kin.mean(0), all_kin.std(0) + 1e-8
    parts = [SpikeWindows(n, behavior=(k - vmean) / vstd, window=args.window) for _, n, k, _ in calib]
    loader = DataLoader(ConcatDataset(parts), batch_size=args.batch, shuffle=True,
                        collate_fn=parts[0].collate, drop_last=True)

    # The world model carries this experiment, so give it depth and weight the forecast.
    model = Noema(dim=args.dim, enc_depth=args.enc_depth, wm_depth=args.wm_depth, heads=8,
                  max_units=cfg.n_channels, behavior_dim=cfg.out_dim, multistep=args.multistep).to(device)

    def log(step, d):
        if step % 1000 == 0 and "loss_forecast" in d:
            print(f"step {step:5d} forecast={d['loss_forecast']:.3f} jepa={d.get('loss_jepa', 0):.3f} "
                  f"behavior={d.get('loss_behavior', 0):.3f}", flush=True)

    train(model, loader, TrainConfig(steps=args.steps, warmup=100, lr=3e-4, w_behavior=args.w_behavior,
                                     w_forecast=args.w_forecast, ckpt=args.ckpt), device=device, on_log=log)
    if args.ckpt:
        print(f"saved {args.ckpt}", flush=True)

    if args.fidelity or not args.sim2real:
        fid = _rollout_fidelity(model, minival, args.window, args.horizon, device)
        m, sm, pers = fid["model"], fid["seed_mean"], fid["persistence"]
        print("rollout population correlation vs horizon, per-session profile removed:",
              flush=True)
        print(f"{'':4}{'h':>3}{'model':>10}{'seed-mean':>12}{'persistence':>14}"
              f"{'over best floor':>18}", flush=True)
        for h in range(len(m)):
            floor = max(sm[h], pers[h])
            print(f"{'':4}{h + 1:>3}{m[h]:>10.3f}{sm[h]:>12.3f}{pers[h]:>14.3f}"
                  f"{m[h] - floor:>+18.3f}", flush=True)
        best = np.maximum(sm, pers)
        print(f"summary: model h1={m[0]:.3f} h{len(m)}={m[-1]:.3f} mean={m.mean():.3f}  |  "
              f"best constant floor mean={best.mean():.3f}  |  "
              f"model over floor mean={(m - best).mean():+.3f}", flush=True)
        if fid["dropped"]:
            print(f"  ({fid['dropped']} degenerate windows dropped)", flush=True)

    if args.sim2real:
        ids = torch.arange(cfg.n_channels)
        stats = (torch.as_tensor(vmean, dtype=torch.float32),
                 torch.as_tensor(vstd, dtype=torch.float32))
        for horizon in (int(h) for h in args.horizons.split(",")):
            tr = [_windows(n, k, m, args.window, horizon, args.stride) for _, n, k, m in calib]
            va = [_windows(n, k, m, args.window, horizon, args.stride) for _, n, k, m in minival]
            cat = lambda parts, i: torch.cat([p[i] for p in parts])
            # Val windows are cut to the same length as the imagined ones so the Gaussian
            # smoothing kernel sees identical trial edges in training and evaluation.
            val_ds = SpikeWindows(cat(va, 1), behavior=cat(va, 2))
            arms, diagnostics = _factorial(model, ids, cat(tr, 0), cat(tr, 1), cat(tr, 2),
                                           val_ds, horizon, args.samples, stats, device)
            print(f"\nsim2real horizon={horizon}  ({cat(tr, 0).size(0)} seeds, "
                  f"{val_ds.heldin.size(0)} eval windows)", flush=True)
            for name, r2 in arms.items():
                print(f"  {name} R2 = {r2:+.4f}", flush=True)
            ceiling = arms["R real spikes / real labels "]
            print(f"  ratio I/R = {arms['I imagined   / head labels '] / ceiling:.3f}   "
                  f"spike cost R-D = {ceiling - arms['D imagined   / real labels ']:+.4f}   "
                  f"label cost R-S = {ceiling - arms['S real spikes / head labels ']:+.4f}", flush=True)
            print("  " + "  ".join(f"{k}={v:.3f}" for k, v in diagnostics.items()), flush=True)


if __name__ == "__main__":
    main()
