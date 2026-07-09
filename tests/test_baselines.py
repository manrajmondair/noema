import torch

from noema.data.dataset import SpikeWindows
from noema.data.synthetic import LinearSpikeSystem
from noema.eval.baselines import gaussian_smooth, ridge_velocity


def test_gaussian_smooth_reduces_temporal_variation():
    x = torch.rand(4, 20, 6)
    y = gaussian_smooth(x, sigma=2.0)
    assert y.shape == x.shape
    rough = lambda t: (t[:, 1:] - t[:, :-1]).abs().mean()
    assert rough(y) < rough(x)  # output is smoother in time than the raw signal


def test_ridge_velocity_baseline_decodes():
    system = LinearSpikeSystem(units=40, latent=6, action_dim=2, seed=1)
    c, _, a, b = system.sample(batch=256, steps=30)
    train = SpikeWindows(c[:200], behavior=b[:200])
    val = SpikeWindows(c[200:], behavior=b[200:])
    assert ridge_velocity(train, val) > 0.2  # a real reference point to beat
