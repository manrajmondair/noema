import torch
from torch.utils.data import DataLoader

from noema import Noema
from noema.data.dataset import SpikeWindows
from noema.data.synthetic import synthetic_batch
from noema.eval.ensemble import ensemble_co_bps
from noema.eval.nlb import evaluate
from noema.train import TrainConfig, train

CPU = torch.device("cpu")


def test_ensemble_not_worse_than_worst_member():
    torch.manual_seed(0)
    counts, _, behavior = synthetic_batch(batch=128, steps=25, units=40, seed=0)
    ds = SpikeWindows(counts[..., :28], counts[..., 28:], behavior)
    loader = DataLoader(ds, batch_size=32, collate_fn=ds.collate, drop_last=True)

    models = []
    for _ in range(2):
        m = Noema(dim=48, enc_depth=2, wm_depth=1, heads=4, max_units=40)
        train(m, loader, TrainConfig(steps=80, warmup=5, lr=3e-3, ckpt=""), device=CPU)
        models.append(m)

    singles = [evaluate(m, ds)["co_bps"] for m in models]
    ensemble = ensemble_co_bps(models, ds, device=CPU)
    assert ensemble >= min(singles) - 0.05  # rate averaging never trails the worst member
