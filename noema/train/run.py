import argparse

import torch
from torch.utils.data import DataLoader

from .. import Noema
from ..data.dataset import SpikeWindows, split_trials
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
                  heads=args.heads, max_units=max_units, sessions=n_sessions, spatial=args.spatial,
                  neuron_mask_ratio=args.neuron_mask, cross=args.cross, attn_pool=args.attn_pool, contrastive=args.contrastive, ssm=args.ssm, ssm_state=args.ssm_state, hybrid=args.hybrid, ssm_dt=args.ssm_dt, film=args.film, graft=args.graft)
    run = wandb_run(args)
    train(model, batches, TrainConfig(steps=args.steps, lr=args.lr, ckpt=args.ckpt), on_log=logger(run))
    if run:
        run.finish()


def fit(args):
    ds = build_dataset(args, "train")
    val_ds = build_dataset(args, "val") if args.dataset == "nlb" else ds
    behavior_dim = ds.behavior.size(-1) if ds.behavior is not None else 0
    max_units = ds.in_ids.numel() + ds.out_ids.numel()
    state = torch.load(args.init, map_location="cpu") if args.init else None
    if state is not None:  # size the table to the pretrained one so its unit rows load
        max_units = max(max_units, state["tokenizer.embed.weight"].shape[0])
    model = Noema(dim=args.dim, enc_depth=args.enc_depth, wm_depth=args.wm_depth,
                  heads=args.heads, max_units=max_units, behavior_dim=behavior_dim, spatial=args.spatial,
                  neuron_mask_ratio=args.neuron_mask, cross=args.cross, attn_pool=args.attn_pool, contrastive=args.contrastive, ssm=args.ssm, ssm_state=args.ssm_state, hybrid=args.hybrid, ssm_dt=args.ssm_dt, film=args.film, graft=args.graft)
    if state is not None:  # warm-start backbone + shared unit embeddings; fresh heads stay fresh
        model.load_state_dict(state, strict=False)

    # Select checkpoints on a set carved from train, not on the reported val split.
    core_ds, select_ds = split_trials(ds, 0.85) if args.dataset == "nlb" else (ds, val_ds)
    loader = DataLoader(core_ds, batch_size=args.batch, shuffle=True,
                        collate_fn=core_ds.collate, drop_last=True)
    run = wandb_run(args)
    train(model, loader, TrainConfig(steps=args.steps, lr=args.lr, ckpt=args.ckpt,
                                     eval_every=args.eval_every, w_contrastive=args.w_contrastive),
          on_log=logger(run), val_ds=select_ds)

    metrics = evaluate(model, val_ds)
    if args.dataset == "nlb":
        from ..eval.nlb import official_velocity_r2
        metrics["official_vel_r2"] = official_velocity_r2(model, ds, val_ds)
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
    p.add_argument("--ckpt", default="checkpoints/noema.pt")
    p.add_argument("--eval-every", type=int, default=500, help="val co-bps checkpoint selection interval")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spatial", action="store_true", help="per-unit spatiotemporal attention (STNDT-style)")
    p.add_argument("--neuron-mask", type=float, default=0.0, help="random-neuron co-smoothing ratio")
    p.add_argument("--cross", action="store_true", help="cross-attention co-smoothing readout (spatial only)")
    p.add_argument("--attn-pool", action="store_true", help="attention pool per-unit tokens into the latent (spatial only)")
    p.add_argument("--contrastive", action="store_true", help="add InfoNCE representation loss (STNDT-style)")
    p.add_argument("--ssm", action="store_true", help="diagonal state-space temporal encoder (S5/LRU-style)")
    p.add_argument("--ssm-state", type=int, default=128, help="SSM state size")
    p.add_argument("--ssm-dt", action="store_true", help="learnable per-mode timescale (S5-style dt)")
    p.add_argument("--hybrid", action="store_true", help="interleave attention layers into the SSM encoder")
    p.add_argument("--film", action="store_true", help="nonlinear FiLM held-out readout (vs linear decode)")
    p.add_argument("--graft", action="store_true", help="GRAFT per-neuron gain interface (read-in gain + per-neuron readout)")
    p.add_argument("--w-contrastive", type=float, default=1.0, help="weight on the contrastive loss")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    pretrain(args) if args.datasets else fit(args)


if __name__ == "__main__":
    main()
