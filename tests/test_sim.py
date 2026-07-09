import torch
from torch.utils.data import DataLoader

from noema import Noema
from noema.data.dataset import SpikeWindows
from noema.data.synthetic import LinearSpikeSystem
from noema.sim import imagine
from noema.train import TrainConfig, train


def _corr(a, b):
    a, b = a.flatten() - a.mean(), b.flatten() - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()


def test_simulator_tracks_ground_truth():
    torch.manual_seed(0)
    cpu = torch.device("cpu")
    system = LinearSpikeSystem(units=50, latent=6, action_dim=2, seed=1)
    counts, unit_ids, actions, behavior = system.sample(batch=256, steps=40)
    ds = SpikeWindows(counts, behavior=behavior, actions=actions)  # all units observed
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=ds.collate, drop_last=True)

    model = Noema(dim=96, enc_depth=3, wm_depth=2, heads=4, max_units=50,
                  action_dim=2, behavior_dim=2)
    train(model, loader, TrainConfig(steps=400, warmup=30, lr=3e-3, ckpt=""), device=cpu)

    # Fresh episode: seed on real spikes, then imagine forward under its own actions.
    c, uid, a, _ = system.sample(batch=32, steps=40)
    seed = 25
    _, true_rates, true_beh = system.rollout(a)
    rates, beh = imagine(model, c[:, :seed], uid, a[:, seed:], seed_actions=a[:, :seed])

    # Compare in the well-scaled log-rate space; raw rates are exp-dominated.
    true_lr = true_rates[:, seed:].clamp_min(1e-6).log()
    null = true_lr.mean(dim=(0, 1), keepdim=True).expand_as(true_lr)

    assert _corr(rates.clamp_min(1e-6).log(), true_lr) > 0.3   # imagined firing tracks truth
    assert _corr(rates.clamp_min(1e-6).log(), true_lr) > _corr(null, true_lr)
    assert _corr(beh, true_beh[:, seed:]) > 0.3                # imagined behavior tracks truth
