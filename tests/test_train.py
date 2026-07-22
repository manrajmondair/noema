import torch
from torch.utils.data import DataLoader

from noema import Noema
from noema.data.dataset import SpikeWindows
from noema.data.synthetic import synthetic_batch
from noema.train import TrainConfig, train


def test_trainer_reduces_loss():
    counts, _, behavior = synthetic_batch(batch=128, steps=30, units=50, behavior_dim=2)
    ds = SpikeWindows(counts[..., :35], counts[..., 35:], behavior)
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=ds.collate, drop_last=True)

    model = Noema(dim=96, enc_depth=2, wm_depth=1, heads=4,
                  max_units=ds.in_ids.numel() + ds.out_ids.numel(), behavior_dim=2)

    history = []
    cfg = TrainConfig(steps=120, warmup=10, lr=3e-3, log_every=1, ckpt="")
    train(model, loader, cfg, device=torch.device("cpu"),
          on_log=lambda s, d: history.append(d["loss_cosmooth"]))

    assert history[-1] < 0.8 * history[0]  # held-out neurons get easier to predict


def test_batch_carries_targets():
    counts, _, behavior = synthetic_batch(batch=8, steps=12, units=20, behavior_dim=2)
    ds = SpikeWindows(counts[..., :14], counts[..., 14:], behavior)
    batch = ds.collate([ds[0], ds[1]])
    assert batch["counts"].shape == (2, 12, 14)
    assert batch["target_counts"].shape == (2, 12, 6)
    assert batch["unit_ids"].tolist() == list(range(14))
    assert batch["target_unit_ids"].tolist() == list(range(14, 20))
