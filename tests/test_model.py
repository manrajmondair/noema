import torch

from noema import Noema
from noema.data.synthetic import synthetic_batch


def test_shapes():
    counts, unit_ids, behavior = synthetic_batch(batch=4, steps=20, units=30, behavior_dim=2)
    model = Noema(dim=64, enc_depth=2, wm_depth=1, heads=4, max_units=32, behavior_dim=2)
    out = model(counts, unit_ids, behavior=behavior)
    assert out["rate"].shape == counts.shape
    assert out["z"].shape == (4, 20, 64)
    assert out["loss_rate"].item() > 0


def test_rollout_shape():
    from noema.sim.rollout import imagine

    counts, unit_ids, _ = synthetic_batch(batch=2, steps=16, units=25, behavior_dim=0)
    model = Noema(dim=64, enc_depth=2, wm_depth=1, heads=4, max_units=32)
    future = torch.zeros(2, 8, 0)  # no actions in the unconditioned model
    rates, behavior = imagine(model, counts, unit_ids, future)
    assert rates.shape == (2, 8, 25)  # imagined future firing rates, seed excluded
    assert behavior is None


def test_multistep_loss_is_opt_in_and_rolls_out():
    counts, unit_ids, _ = synthetic_batch(batch=8, steps=20, units=30)
    actions = torch.randn(8, 20, 2)
    # default: no multi-step term
    off = Noema(dim=48, enc_depth=2, wm_depth=2, heads=4, max_units=32, action_dim=2)
    assert "loss_multistep" not in off(counts, unit_ids, actions=actions)
    # opt-in: present and finite, both action-conditioned and unconditioned
    on = Noema(dim=48, enc_depth=2, wm_depth=2, heads=4, max_units=32, action_dim=2, multistep=6)
    for a in (actions, None):
        ms = on(counts, unit_ids, actions=a)["loss_multistep"]
        assert torch.isfinite(ms) and ms.item() > 0


def test_ssm_kernel_matches_sequential_and_reconstructs():
    from noema.eval.ensemble_run import build_from_state
    from noema.models.ssm import DiagonalSSM

    # the parallel materialized-kernel form must equal the sequential recurrence
    torch.manual_seed(0)
    m = DiagonalSSM(dim=24, state=48)
    u = torch.randn(3, 18, 24)
    assert torch.allclose(m(u), m.sequential(u), atol=1e-4)

    # an SSM-encoder Noema builds, runs, and round-trips through build_from_state
    counts, unit_ids, _ = synthetic_batch(batch=4, steps=16, units=30)
    hi, in_ids, out_ids = counts[..., :20], unit_ids[:20], unit_ids[20:]
    model = Noema(dim=48, enc_depth=2, wm_depth=1, heads=4, max_units=32, ssm=True)
    model.eval()
    rebuilt, desc = build_from_state(model.state_dict(), max_units=32, heads=4)
    assert "ssm" in desc
    rebuilt.eval()
    assert torch.allclose(model.cosmooth(hi, in_ids, out_ids), rebuilt.cosmooth(hi, in_ids, out_ids), atol=1e-5)


def test_film_readout_nonlinear_and_reconstructs():
    from noema.eval.ensemble_run import build_from_state

    counts, unit_ids, _ = synthetic_batch(batch=4, steps=16, units=30)
    hi, in_ids, out_ids = counts[..., :20], unit_ids[:20], unit_ids[20:]
    m = Noema(dim=48, enc_depth=2, wm_depth=1, heads=4, max_units=32, ssm=True, film=True)
    m.eval()
    # the FiLM co-smoothing readout is genuinely nonlinear (differs from the linear decode)
    z = m.encode(hi, in_ids)
    assert not torch.allclose(m.cosmooth(hi, in_ids, out_ids), m.tokenizer.decode(z, out_ids), atol=1e-3)
    # and round-trips through build_from_state (film auto-detected from weights)
    r, _ = build_from_state(m.state_dict(), max_units=32, heads=4)
    r.eval()
    assert torch.allclose(m.cosmooth(hi, in_ids, out_ids), r.cosmooth(hi, in_ids, out_ids), atol=1e-5)


def test_contrastive_loss_is_opt_in_and_finite():
    counts, unit_ids, _ = synthetic_batch(batch=6, steps=18, units=30)
    off = Noema(dim=48, enc_depth=2, wm_depth=1, heads=4, max_units=32)
    assert "loss_contrastive" not in off(counts, unit_ids)
    on = Noema(dim=48, enc_depth=2, wm_depth=1, heads=4, max_units=32, contrastive=True)
    lc = on(counts, unit_ids)["loss_contrastive"]
    assert torch.isfinite(lc) and lc.item() > 0


def test_attention_pool_reconstructs_and_differs_from_mean():
    from noema.eval.ensemble_run import build_from_state

    counts, unit_ids, _ = synthetic_batch(batch=4, steps=16, units=30, behavior_dim=0)
    hi = counts[..., :20]  # held-in spikes; the other 10 units are the co-smoothing target
    in_ids, out_ids = unit_ids[:20], unit_ids[20:]
    model = Noema(dim=48, enc_depth=2, wm_depth=1, heads=4, max_units=32, spatial=True, attn_pool=True)
    model.eval()

    # the pooled latent must actually use the learned query, not reduce to a mean
    tokens = model.encoder(model.tokenizer.encode_units(hi, in_ids))
    assert not torch.allclose(model.pooler(tokens), tokens.mean(2), atol=1e-4)

    # build_from_state auto-detects the pooler from the weights and rebuilds bit-identically
    rebuilt, desc = build_from_state(model.state_dict(), max_units=32, heads=4)
    assert "attnpool" in desc
    rebuilt.eval()
    a = model.cosmooth(hi, in_ids, out_ids)
    b = rebuilt.cosmooth(hi, in_ids, out_ids)
    assert torch.allclose(a, b, atol=1e-5)


def test_overfits_single_batch():
    torch.manual_seed(0)
    counts, unit_ids, behavior = synthetic_batch(batch=16, steps=40, units=50, behavior_dim=2)
    model = Noema(dim=128, enc_depth=3, wm_depth=2, heads=4, max_units=64, behavior_dim=2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first = None
    for step in range(300):
        out = model(counts, unit_ids, behavior=behavior)
        loss = out["loss_rate"] + out["loss_jepa"] + 5 * out["loss_behavior"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_teacher()
        if step == 0:
            first = (out["loss_rate"].item(), out["loss_behavior"].item())

    assert out["loss_rate"].item() < 0.6 * first[0]      # firing rates fit
    assert out["loss_behavior"].item() < 0.2 * first[1]  # behavior decodes
