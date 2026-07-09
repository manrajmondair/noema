import torch
from torch.utils.data import DataLoader

from noema import Noema
from noema.data.dataset import SpikeWindows
from noema.data.synthetic import SensorySystem
from noema.train import TrainConfig, train


def _corr(a, b):
    a, b = a.flatten() - a.mean(), b.flatten() - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()


def test_predicts_response_to_stimulus():
    torch.manual_seed(0)
    cpu = torch.device("cpu")
    system = SensorySystem(units=50, latent=6, stim_dim=8, seed=1)
    counts, unit_ids, stim, behavior = system.sample(batch=256, steps=40)
    ds = SpikeWindows(counts, behavior=behavior, context=stim)
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=ds.collate, drop_last=True)

    model = Noema(dim=96, enc_depth=2, wm_depth=2, heads=4, max_units=50,
                  context_dim=8, behavior_dim=2)
    train(model, loader, TrainConfig(steps=350, warmup=30, lr=3e-3, ckpt=""), device=cpu)

    stimulus = system.sample(batch=32, steps=40)[2]
    true_rates, _ = system.response(stimulus)
    pred = model.predict_response(stimulus, unit_ids).clamp_min(1e-6).log()
    mismatch = model.predict_response(stimulus[torch.randperm(32)], unit_ids).clamp_min(1e-6).log()
    truth = true_rates.clamp_min(1e-6).log()

    assert _corr(pred, truth) > 0.3               # tracks the true neural response
    assert _corr(pred, truth) > _corr(mismatch, truth)  # genuinely uses stimulus content
