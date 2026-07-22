import math

import torch

from noema import Noema
from noema.data.dataset import SpikeWindows
from noema.data.synthetic import MultiSessionSystem
from noema.eval.calibration import calibration_curve


def _session(system, s, batch):
    c, _, a, b = system.sample(s, batch=batch, steps=20)
    return SpikeWindows(c, behavior=b, actions=a, unit_ids=system.unit_ids(s))


def test_calibration_curve_reports_both_methods_per_budget():
    torch.manual_seed(0)
    system = MultiSessionSystem(sessions=2, units=20, latent=5, seed=1)
    pretrained = Noema(dim=48, enc_depth=1, wm_depth=1, heads=4, max_units=40,
                       action_dim=2, behavior_dim=2)
    fresh = lambda: Noema(dim=48, enc_depth=1, wm_depth=1, heads=4, max_units=40,
                          action_dim=2, behavior_dim=2)

    curve = calibration_curve(pretrained, fresh, _session(system, 1, 16), _session(system, 1, 32),
                              budgets=[4, 12], adapt_steps=40, scratch_steps=40,
                              device=torch.device("cpu"))

    assert [pt["trials"] for pt in curve] == [4, 12]
    for pt in curve:
        assert math.isfinite(pt["transfer"]) and math.isfinite(pt["scratch"])
