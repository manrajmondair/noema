import torch
from torch.utils.data import DataLoader

from noema import Noema
from noema.data.dataset import SpikeWindows, split_trials
from noema.data.synthetic import synthetic_batch
from noema.eval.nlb import _raw_behavior, evaluate
from noema.train import TrainConfig, train


def test_behavior_stats_recover_raw_kinematics():
    # the leaderboard R² scores against raw velocity; check we can undo the per-split
    # standardization and that split_trials carries the stats through
    raw = torch.randn(20, 5, 2) * 3.0 + 1.0
    flat = raw.reshape(-1, 2)
    mean, std = flat.mean(0), flat.std(0) + 1e-6
    standardized = (raw - mean) / std
    ds = SpikeWindows(torch.rand(20, 5, 8), torch.rand(20, 5, 3), standardized,
                      behavior_stats=(mean, std))

    core, sel = split_trials(ds, 0.7)
    assert core.behavior_stats is not None and sel.behavior_stats is not None  # propagated

    recovered = _raw_behavior(standardized.reshape(-1, 2).numpy(), ds)
    assert torch.allclose(torch.as_tensor(recovered, dtype=torch.float32), flat, atol=1e-4)

    v = standardized.reshape(-1, 2).numpy()
    assert (_raw_behavior(v, SpikeWindows(torch.rand(20, 5, 8), None, standardized)) == v).all()


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
