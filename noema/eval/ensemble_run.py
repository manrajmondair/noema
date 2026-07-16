"""Score a rate-space ensemble of trained checkpoints on an NLB dataset.

Each checkpoint's architecture (width, depth, spatial-or-temporal) is inferred
from its weights, so temporal, spatial, and variant models can be ensembled
together — a diverse ensemble beats a same-architecture one.
"""

import argparse

import torch

from .. import Noema
from ..data.nlb import load_nlb
from ..utils import default_device
from .ensemble import ensemble_co_bps


def build_from_state(state, max_units, heads=8):
    dim = state["tokenizer.embed.weight"].shape[1]
    spatial = any(k.startswith("encoder.spatial.") for k in state)
    prefix = "encoder.temporal." if spatial else "encoder.blocks."
    depth = len({k[len(prefix):].split(".")[0] for k in state if k.startswith(prefix)})
    model = Noema(dim=dim, enc_depth=depth, wm_depth=1, heads=heads,
                  max_units=max_units, spatial=spatial)
    model.load_state_dict(state, strict=False)  # world model unused here; heads is not in weights
    return model, f"dim{dim} depth{depth} {'spatial' if spatial else 'temporal'}"


def main():
    p = argparse.ArgumentParser(prog="noema.eval.ensemble_run")
    p.add_argument("--ckpts", required=True, help="comma-separated checkpoint paths")
    p.add_argument("--name", default="mc_maze")
    p.add_argument("--path", required=True)
    p.add_argument("--bin-ms", type=int, default=5)
    p.add_argument("--heads", type=int, default=8)
    args = p.parse_args()

    val = load_nlb(args.path, args.name, args.bin_ms, split="val")
    max_units = val.in_ids.numel() + val.out_ids.numel()
    device = default_device()

    models = []
    for path in args.ckpts.split(","):
        model, desc = build_from_state(torch.load(path, map_location="cpu"), max_units, args.heads)
        models.append(model.to(device))
        print(f"  {path.split('/')[-1]}: {desc}", flush=True)

    best = (0.0, -1e9)
    for sigma in (0.0, 1.0, 1.5, 2.0, 2.5, 3.0):
        cobps = ensemble_co_bps(models, val, device=device, smooth=sigma)
        print(f"  smooth={sigma}: co_bps = {cobps:.4f}", flush=True)
        best = max(best, (sigma, cobps), key=lambda x: x[1])
    print(f"ensemble co_bps ({len(models)} members) = {best[1]:.4f} (smooth={best[0]})", flush=True)


if __name__ == "__main__":
    main()
