import argparse

import torch
from torch.utils.data import DataLoader

from .. import Noema
from ..data.dataset import SpikeWindows
from ..data.pretrain import combine_sessions
from ..data.synthetic import synthetic_batch
from ..eval.nlb import evaluate
from .trainer import TrainConfig, train


def build_dataset(args, split="train"):
    if args.dataset == "nlb":
        from ..data.nlb import load_nlb
        return load_nlb(args.path, args.name, args.bin_ms, args.window, split=split)
    counts, _, behavior = synthetic_batch(batch=512, steps=40, units=80, behavior_dim=2)
    return SpikeWindows(counts[..., :60], counts[..., 60:], behavior)


def logger(run):
    def log(step, losses):
        print(f"step {step:>6} " + " ".join(f"{k[5:]}={v:.4f}" for k, v in losses.items()), flush=True)
        if run:
            run.log(losses, step=step)
    return log


def wandb_run(args):
    if not args.wandb:
        return None
    import wandb
    return wandb.init(project="noema", config=vars(args))


def pretrain(args):
    """Self-supervised multi-session pretraining — the cross-subject Stage 1."""
    from ..data.nlb import load_nlb
    names = [n for n in args.datasets.split(",") if n]
    sessions = [load_nlb(f"{args.data_root}/{n}", n, args.bin_ms, args.window) for n in names]
    batches, max_units, n_sessions = combine_sessions(sessions, args.batch)

    model = Noema(dim=args.dim, enc_depth=args.enc_depth, wm_depth=args.wm_depth,
                  heads=args.heads, max_units=max_units, sessions=n_sessions)
    run = wandb_run(args)
    train(model, batches, TrainConfig(steps=args.steps, lr=args.lr), on_log=logger(run))
    if run:
        run.finish()


def fit(args):
    ds = build_dataset(args, "train")
    val_ds = build_dataset(args, "val") if args.dataset == "nlb" else ds
    behavior_dim = ds.behavior.size(-1) if ds.behavior is not None else 0
    max_units = ds.in_ids.numel() + ds.out_ids.numel()
    model = Noema(dim=args.dim, enc_depth=args.enc_depth, wm_depth=args.wm_depth,
                  heads=args.heads, max_units=max_units, behavior_dim=behavior_dim)
    if args.init:  # warm-start the shared backbone; fresh heads stay fresh
        model.load_state_dict(torch.load(args.init, map_location="cpu"), strict=False)

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        collate_fn=ds.collate, drop_last=True)
    run = wandb_run(args)
    train(model, loader, TrainConfig(steps=args.steps, lr=args.lr), on_log=logger(run))

    metrics = evaluate(model, val_ds)
    print("eval " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
    if run:
        run.summary.update(metrics)
        run.finish()


def main():
    p = argparse.ArgumentParser(prog="noema.train")
    p.add_argument("--dataset", choices=["nlb", "synthetic"], default="synthetic")
    p.add_argument("--datasets", help="comma-separated NLB names for multi-session pretraining")
    p.add_argument("--data-root", default="data")
    p.add_argument("--init", help="checkpoint to warm-start the backbone from")
    p.add_argument("--path")
    p.add_argument("--name", default="mc_maze")
    p.add_argument("--bin-ms", type=int, default=5)
    p.add_argument("--window", type=int)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--enc-depth", type=int, default=6)
    p.add_argument("--wm-depth", type=int, default=3)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    pretrain(args) if args.datasets else fit(args)


if __name__ == "__main__":
    main()
