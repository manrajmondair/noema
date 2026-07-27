import math
import os
from dataclasses import dataclass

import torch

from ..utils import default_device


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    steps: int = 20_000
    warmup: int = 500
    grad_clip: float = 1.0
    w_rate: float = 1.0
    w_cosmooth: float = 1.0
    w_ncosmooth: float = 1.0
    w_jepa: float = 1.0
    w_forecast: float = 1.0
    w_multistep: float = 1.0
    w_contrastive: float = 1.0
    w_behavior: float = 1.0
    w_session: float = 1.0
    w_sensory: float = 1.0
    amp: bool = True
    log_every: int = 50
    eval_every: int = 0  # >0 with a val set: keep the best val co-bps checkpoint, not the last
    ckpt: str = "checkpoints/noema.pt"


def _lr_scale(step, cfg):
    if step < cfg.warmup:
        return step / max(1, cfg.warmup)
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _endless(loader):
    while True:
        yield from loader


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _save(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def train(model, loader, cfg, device=None, on_log=None, val_ds=None):
    device = device or default_device()
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp = cfg.amp and device.type == "cuda"  # bf16: wide dynamic range, no loss scaler needed

    select = val_ds is not None and cfg.eval_every > 0
    best = None
    batches = _endless(loader)
    for step in range(cfg.steps):
        batch = _to_device(next(batches), device)
        for group in opt.param_groups:
            group["lr"] = cfg.lr * _lr_scale(step, cfg)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(
                batch["counts"], batch["unit_ids"],
                actions=batch.get("actions"),
                behavior=batch.get("behavior"),
                target_counts=batch.get("target_counts"),
                target_unit_ids=batch.get("target_unit_ids"),
                session=batch.get("session"),
                context=batch.get("context"),
            )
            loss = (cfg.w_rate * out["loss_rate"]
                    + cfg.w_jepa * out["loss_jepa"]
                    + cfg.w_forecast * out["loss_forecast"]
                    + cfg.w_multistep * out.get("loss_multistep", 0.0)
                    + cfg.w_contrastive * out.get("loss_contrastive", 0.0)
                    + cfg.w_cosmooth * out.get("loss_cosmooth", 0.0)
                    + cfg.w_ncosmooth * out.get("loss_ncosmooth", 0.0)
                    + cfg.w_behavior * out.get("loss_behavior", 0.0)
                    + cfg.w_session * out.get("loss_session", 0.0)
                    + cfg.w_sensory * out.get("loss_sensory", 0.0))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        model.update_teacher()

        if on_log and step % cfg.log_every == 0:
            on_log(step, {k: v.detach().item() for k, v in out.items() if k.startswith("loss")})

        if select and step > 0 and step % cfg.eval_every == 0:
            from ..eval.nlb import evaluate
            cobps = evaluate(model, val_ds, device=device).get("co_bps", float("-inf"))
            model.train()
            if on_log:
                on_log(step, {"loss_val_cobps": cobps})
            if best is None or cobps > best:
                best = cobps
                if cfg.ckpt:
                    _save(model, cfg.ckpt)

    if select and best is not None and cfg.ckpt:
        model.load_state_dict(torch.load(cfg.ckpt, map_location=device))  # restore best-val weights
    elif cfg.ckpt:
        _save(model, cfg.ckpt)
    return model
