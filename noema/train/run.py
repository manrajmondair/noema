import argparse

from torch.utils.data import DataLoader

from .. import Noema
from ..data.dataset import SpikeWindows
from ..data.synthetic import synthetic_batch
from .trainer import TrainConfig, train


def build_dataset(args):
    if args.dataset == "nlb":
        from ..data.nlb import load_nlb
        return load_nlb(args.path, args.name, args.bin_ms, args.window)
    counts, _, behavior = synthetic_batch(batch=512, steps=40, units=80, behavior_dim=2)
    return SpikeWindows(counts[..., :60], counts[..., 60:], behavior)


def main():
    p = argparse.ArgumentParser(prog="noema.train")
    p.add_argument("--dataset", choices=["nlb", "synthetic"], default="synthetic")
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

    ds = build_dataset(args)
    behavior_dim = ds.behavior.size(-1) if ds.behavior is not None else 0
    max_units = ds.in_ids.numel() + ds.out_ids.numel()
    model = Noema(dim=args.dim, enc_depth=args.enc_depth, wm_depth=args.wm_depth,
                  heads=args.heads, max_units=max_units, behavior_dim=behavior_dim)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        collate_fn=ds.collate, drop_last=True)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="noema", config=vars(args))

    def log(step, losses):
        print(f"step {step:>6} " + " ".join(f"{k[5:]}={v:.4f}" for k, v in losses.items()), flush=True)
        if run:
            run.log(losses, step=step)

    train(model, loader, TrainConfig(steps=args.steps, lr=args.lr), on_log=log)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
