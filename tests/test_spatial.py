import torch
from torch.utils.data import DataLoader

from noema import Noema
from noema.data.dataset import SpikeWindows
from noema.data.synthetic import synthetic_batch
from noema.train import TrainConfig, train


def test_spatial_model_trains_and_rate_head_shape():
    torch.manual_seed(0)
    counts, _, behavior = synthetic_batch(batch=64, steps=28, units=40, behavior_dim=2, seed=0)
    ds = SpikeWindows(counts[..., :28], counts[..., 28:], behavior)
    loader = DataLoader(ds, batch_size=16, collate_fn=ds.collate, drop_last=True)
    model = Noema(dim=48, enc_depth=2, wm_depth=1, heads=4, max_units=40,
                  behavior_dim=2, spatial=True)

    logs = []
    train(model, loader, TrainConfig(steps=60, warmup=5, lr=3e-3, ckpt=""),
          device=torch.device("cpu"), on_log=lambda s, d: logs.append(d))
    assert logs[-1]["loss_rate"] < 0.7 * logs[0]["loss_rate"]  # per-unit path learns

    model.eval()
    b = next(iter(loader))
    out = model(b["counts"], b["unit_ids"], target_counts=b.get("target_counts"),
                target_unit_ids=b.get("target_unit_ids"))
    assert out["rate"].shape == b["counts"].shape        # per-unit rate for observed units
    assert out["z"].shape[:2] == b["counts"].shape[:2]   # pooled latent [B,T,dim]
