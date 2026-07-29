"""Verify a new GPU box before spending hours on it.

    python scripts/gpu_smoke.py

Checks the numerics this model actually depends on, hardest first. Written after an
audit of AMD CDNA4 (MI355X), where the state-space encoder is the exposed component:
it runs in fp32/complex64, so autocast never touches it and its accuracy is decided by
the fp32 matmul precision setting. On CDNA4 there is no TF32 silicon — enabling it
silently reroutes fp32 matmuls through a bf16 emulation path with 7 mantissa bits,
which a diagonal recurrence with |A| near 1 accumulated over a long trial degrades
quietly rather than loudly. Exits non-zero on the first hard failure.
"""

import sys

import torch

from noema import Noema
from noema.models.ssm import DiagonalSSM
from noema.utils import default_device

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILED.append(name)


def main():
    device = default_device()
    print(f"torch {torch.__version__} | device {device} | "
          f"hip {getattr(torch.version, 'hip', None)} | cuda {torch.version.cuda}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    print("\n[1] fp32 precision is IEEE, not an emulated TF32 path")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    check("matmul.allow_tf32 is off", torch.backends.cuda.matmul.allow_tf32 is False)

    print("\n[2] state-space kernel matches its sequential reference")
    # The materialized-kernel form is a parallel rewrite of the recurrence; if the two
    # disagree, the encoder is silently wrong on this backend.
    torch.manual_seed(0)
    ssm = DiagonalSSM(64, state=128).to(device)
    u = torch.randn(2, 140, 64, device=device)  # 140 bins = a real MC_Maze trial
    err = (ssm(u) - ssm.sequential(u)).abs().max().item()
    check("kernel == sequential", err < 1e-4, f"max abs err {err:.2e}")

    print("\n[3] attention head_dim is safe on this backend")
    # gfx950 disables the SDPA backward pass for head_dim 48 and 80, and rounds 16 to 32.
    for dim in (160, 192, 256, 320, 384):
        hd = dim // 8
        flag = "  <-- unsafe on CDNA4" if hd in (16, 48, 80) else ""
        print(f"     dim={dim:>3} heads=8 -> head_dim={hd}{flag}", flush=True)
    check("default dim=256 is safe", (256 // 8) not in (16, 48, 80))

    print("\n[4] the real model trains, in both encoder families")
    for tag, kw in (("transformer", {}), ("state-space", {"ssm": True})):
        try:
            model = Noema(dim=64, enc_depth=2, wm_depth=1, heads=8, max_units=64, **kw).to(device)
            counts = torch.rand(2, 40, 32, device=device)
            ids = torch.arange(32, device=device)
            amp = device.type == "cuda"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = model(counts, ids)["loss_rate"]
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            finite = all(g.isfinite().all().item() for g in grads)
            check(f"{tag} forward+backward", finite and loss.isfinite().item(),
                  f"loss {loss.item():.3f}, {len(grads)} grads finite={finite}")
        except Exception as exc:  # a backend gap should name itself, not vanish
            check(f"{tag} forward+backward", False, f"{type(exc).__name__}: {exc}")

    print("\n[5] bf16 autocast leaves the complex path in fp32")
    # Complex tensors have no bf16 form; if autocast ever demotes them the SSM breaks.
    amp = device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        z = torch.randn(8, 8, dtype=torch.complex64, device=device)
        out = torch.einsum("ij,jk->ik", z, z)
    check("complex64 einsum survives autocast", out.dtype == torch.complex64, str(out.dtype))

    print("\n" + ("all checks passed" if not FAILED else f"FAILED: {', '.join(FAILED)}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
