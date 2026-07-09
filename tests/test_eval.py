import torch
from torch.utils.data import DataLoader

from noema import Noema
from noema.data.dataset import SpikeWindows
from noema.data.synthetic import synthetic_batch
from noema.eval.nlb import evaluate
from noema.train import TrainConfig, train


def test_model_beats_null_on_heldout_after_training():
    cpu = torch.device("cpu")
    counts, _, behavior = synthetic_batch(batch=256, steps=30, units=60, behavior_dim=2)
    ds = SpikeWindows(counts[..., :40], counts[..., 40:], behavior)
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=ds.collate, drop_last=True)

    model = Noema(dim=96, enc_depth=2, wm_depth=1, heads=4, max_units=60, behavior_dim=2)
    train(model, loader, TrainConfig(steps=250, warmup=20, lr=3e-3, ckpt=""), device=cpu)

    metrics = evaluate(model, ds, device=cpu)
    assert metrics["co_bps"] > 0.0   # held-out neurons predicted better than their mean
    assert metrics["vel_r2"] > 0.0   # behavior decodes above the mean baseline
