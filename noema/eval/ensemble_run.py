"""Score a rate-space ensemble of trained checkpoints on an NLB dataset."""

import argparse

import torch

from .. import Noema
from ..data.nlb import load_nlb
from ..utils import default_device
from .ensemble import ensemble_co_bps


def main():
    p = argparse.ArgumentParser(prog="noema.eval.ensemble_run")
    p.add_argument("--ckpts", required=True, help="comma-separated checkpoint paths")
    p.add_argument("--name", default="mc_maze")
    p.add_argument("--path", required=True)
    p.add_argument("--bin-ms", type=int, default=5)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--enc-depth", type=int, default=6)
    p.add_argument("--wm-depth", type=int, default=3)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--spatial", action="store_true")
    args = p.parse_args()

    val = load_nlb(args.path, args.name, args.bin_ms, split="val")
    max_units = val.in_ids.numel() + val.out_ids.numel()
    device = default_device()

    models = []
    for path in args.ckpts.split(","):
        model = Noema(dim=args.dim, enc_depth=args.enc_depth, wm_depth=args.wm_depth,
                      heads=args.heads, max_units=max_units, spatial=args.spatial)
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        models.append(model.to(device))

    cobps = ensemble_co_bps(models, val, device=device)
    print(f"ensemble co_bps ({len(models)} members) = {cobps:.4f}", flush=True)


if __name__ == "__main__":
    main()
