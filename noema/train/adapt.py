"""Cross-session few-shot adaptation.

Only the per-unit embeddings are trained, so a new session's units learn to
route into the frozen, pretrained latent dynamics. Because embedding gradients
are sparse, this touches only the rows for the units actually present.
"""

import torch

from ..utils import default_device


def few_shot_adapt(model, loader, steps=200, lr=3e-3, device=None):
    device = device or default_device()
    model.to(device).train()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.tokenizer.parameters():
        p.requires_grad_(True)
    # No weight decay: rows for absent units get zero gradient and must stay put.
    opt = torch.optim.AdamW(model.tokenizer.parameters(), lr=lr, weight_decay=0.0)

    batches = iter(loader)
    for _ in range(steps):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["counts"], batch["unit_ids"], actions=batch.get("actions"),
                    behavior=batch.get("behavior"))
        # Calibration is supervised: align the new units to the frozen decoder via
        # the behavior labels, with reconstruction as a light regularizer.
        recon = out["loss_rate"] + out["loss_forecast"]
        loss = out["loss_behavior"] + 0.2 * recon if "loss_behavior" in out else recon
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.requires_grad_(True)
    return model
