import torch

from noema.eval.metrics import bits_per_spike, r2_score


def test_bits_per_spike_rewards_true_rates():
    torch.manual_seed(0)
    rates = torch.rand(200, 20, 8) * 2 + 0.1  # ground-truth lambda
    spikes = torch.poisson(rates)
    mean = spikes.mean(dim=(0, 1), keepdim=True).expand_as(spikes)

    assert bits_per_spike(rates, spikes) > 0          # true rates beat the null
    assert abs(bits_per_spike(mean, spikes)) < 1e-5   # the null scores ~0


def test_r2_perfect_and_null():
    target = torch.randn(100, 2)
    assert r2_score(target, target) > 0.999
    assert r2_score(target.mean(0).expand_as(target), target) < 1e-4
