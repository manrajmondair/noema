"""Score a rate-space ensemble of trained checkpoints on an NLB dataset.

Each checkpoint's architecture (width, depth, spatial-or-temporal) is inferred
from its weights, so temporal, spatial, and variant models can be ensembled
together into a diverse ensemble.
"""

import argparse

import torch

from .. import Noema
from ..data.dataset import split_trials
from ..data.nlb import load_nlb
from ..utils import default_device
from .baselines import gaussian_smooth
from .ensemble import greedy_ensemble, member_rates
from .metrics import bits_per_spike


def build_from_state(state, max_units, heads=8):
    dim = state["tokenizer.embed.weight"].shape[1]
    if "num_heads" in state:  # checkpoints self-describe their head count; older ones fall back
        heads = int(state["num_heads"].item())
    spatial = any(k.startswith("encoder.spatial.") for k in state)
    cross = any(k.startswith("cross.") for k in state)
    attn_pool = any(k.startswith("pooler.") for k in state)
    ssm = any(".ssm." in k for k in state)
    graft = any(k.startswith("tokenizer.gin.") for k in state)
    film = any(k.startswith("film.") for k in state)
    hybrid = ssm and any(k.startswith("encoder.blocks.") and ".attn." in k for k in state)
    ssm_state = next((state[k].shape[0] for k in state if k.endswith(".ssm.fwd.B")), 128)
    ssm_dt = any(k.endswith(".ssm.fwd.log_dt") for k in state)
    prefix = "encoder.temporal." if spatial else "encoder.blocks."
    depth = len({k[len(prefix):].split(".")[0] for k in state if k.startswith(prefix)})
    model = Noema(dim=dim, enc_depth=depth, wm_depth=1, heads=heads,
                  max_units=max_units, spatial=spatial, cross=cross, attn_pool=attn_pool, ssm=ssm, ssm_state=ssm_state, hybrid=hybrid, ssm_dt=ssm_dt, film=film, graft=graft)
    model.load_state_dict(state, strict=False)  # world model unused here; heads is not in weights
    tag = "spatial+cross" if cross else ("spatial" if spatial else ("hybrid" if hybrid else "ssm") if ssm else "temporal")
    if attn_pool:
        tag += "+attnpool"
    return model, f"dim{dim} depth{depth} {tag}"


def main():
    p = argparse.ArgumentParser(prog="noema.eval.ensemble_run")
    p.add_argument("--ckpts", required=True, help="comma-separated checkpoint paths")
    p.add_argument("--name", default="mc_maze")
    p.add_argument("--path", required=True)
    p.add_argument("--bin-ms", type=int, default=5)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mint", action="store_true", help="add the MINT trajectory-library member (decorrelated)")
    p.add_argument("--tta", type=int, default=0, help="test-time augmentation: average k masked passes per member")
    args = p.parse_args()

    val = load_nlb(args.path, args.name, args.bin_ms, split="val")
    _, select = split_trials(load_nlb(args.path, args.name, args.bin_ms, split="train"), 0.85)
    max_units = val.in_ids.numel() + val.out_ids.numel()
    device = default_device()

    models = []
    for path in args.ckpts.split(","):
        model, desc = build_from_state(torch.load(path, map_location="cpu"), max_units, args.heads)
        models.append(model.to(device))
        print(f"  {path.split('/')[-1]}: {desc}", flush=True)

    # Greedy-select members and tune smoothing on the held-out set; report once on val.
    sel_r, sel_t = member_rates(models, select, device=device, tta=args.tta)
    val_r, val_t = member_rates(models, val, device=device, tta=args.tta)

    # MINT: a decorrelated (non-NN) member. Its rates are built on the same core/select/val
    # trials the transformers use (mint_member_rates replicates split_trials seed 0), so they
    # append directly to the greedy pool. Assert trial alignment via the shared targets.
    if args.mint:
        import numpy as np

        from .mint import mint_member_rates
        m_sel, m_sel_t, m_val, m_val_t = mint_member_rates(args.path, args.name, args.bin_ms)
        assert np.allclose(m_sel_t, sel_t.numpy()) and np.allclose(m_val_t, val_t.numpy()), \
            "MINT/transformer trial misalignment — check split_trials seed/frac"
        sel_r.append(torch.as_tensor(m_sel, dtype=torch.float32))
        val_r.append(torch.as_tensor(m_val, dtype=torch.float32))
        print("added MINT member (decorrelated trajectory library)", flush=True)

    chosen = greedy_ensemble(sel_r, sel_t)
    sel_avg = sum(sel_r[j] for j in chosen) / len(chosen)
    val_avg = sum(val_r[j] for j in chosen) / len(chosen)
    sigmas = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0)
    sigma = max(sigmas, key=lambda s: bits_per_spike(gaussian_smooth(sel_avg, s) if s else sel_avg, sel_t))
    cobps = bits_per_spike(gaussian_smooth(val_avg, sigma) if sigma else val_avg, val_t)
    print(f"greedy picked {len(chosen)} of {len(models)} members", flush=True)
    print(f"ensemble co_bps (val) = {cobps:.4f}  (smooth={sigma}, greedy+sigma tuned on held-out)", flush=True)


if __name__ == "__main__":
    main()
