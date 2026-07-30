# Noema

**A world model of neural population dynamics.** Noema treats a neural population as a dynamical
system and learns to predict its own future — `z_{t+1} ~ P(· | z_t, action_t, context_t)` over a
learned latent `z` — so decoding, few-shot calibration, and closed-loop simulation follow from one
objective.

## Results

Official scorers, real recordings. co-bps is measured against the public NLB test labels
(`nlb_tools`) — the leaderboard metric and split.

**MC_Maze — co-smoothing bits/spike (test)**

| Method | co-bps |
|---|---:|
| GRAFT (ensemble) | 0.387 |
| STNDT (ensemble) | 0.386 |
| S5 | 0.382 |
| STNDT (single) | 0.369 |
| **Noema** | **0.367** |
| AutoLFADS | 0.336 |
| MINT | 0.330 |
| NDT | 0.323 |

Above the classic baselines, within 0.02 of the top ensembles. The gap is information-bound:
0.367 is 94% of the 0.392 known-condition oracle, and no reweighting, stronger single, or dedicated
condition classifier sharpens the shared posterior. What remains is `H(condition | held-in)` —
trials whose observed neurons do not determine the reach ([details](scripts/nlb_submission.md)).

**Forward model & transfer** (real data)

FALCON H1 cross-session zero-shot velocity R² is **~0.6**: train on ten held-in sessions, decode
unseen sessions from the same subject on different recording days. The splits are session-disjoint,
so this measures the electrode drift that the benchmark exists to test.

Held-in velocity R² on the local minival split, re-measured with the overlap excised, is
**0.545 ± 0.006** (window 50 / enc 4) and **0.603 ± 0.003** (window 75 / enc 5), three seeds each,
scored by the official `FalconEvaluator`. The figures previously reported for these configurations
were 0.768 and 0.871: in this dandiset each `held-in-minival` recording is a byte-identical prefix
of the matching `held-in-calib` recording, so a run that fits on calibration and scores minival is
scoring its own training data. `disjoint_calib` in [`noema/eval/falcon.py`](noema/eval/falcon.py)
excises it. The cost was not uniform — 0.22 at the weaker configuration and 0.27 at the stronger —
and the longer-window advantage halves once it is removed, from +0.10 to +0.06.

These are a dev split, not the leaderboard: minival is public and was the split the configuration
sweep selected on, so the numbers carry adaptive optimism on top of everything above. The
leaderboard figure needs an EvalAI submission against sequestered held-out labels. The ± is
dispersion across the 13 minival sessions.

**The world-model rollout figures are retired, not corrected.** Reported as raw population
correlation ~0.48 decaying to 0.21, or holding flat at 0.45 under the multi-step objective, they
measured the static per-channel firing profile rather than prediction — a forecast emitting nothing
but each channel's average scores ~0.5 on that metric. Re-measured with the profile removed per
session and the published objective restored (no behavior head), the rollout reaches 0.12 at one
step and 0.06 by ten, against **0.14 and 0.10 for a strictly causal forecast that averages the seed
window and holds it flat**. The model does not beat that floor at any horizon, under either
objective. Training on the overlap changes this by less than seed noise, so contamination was never
what inflated these figures — the metric was. The multi-step objective is still worth ~0.07 mean
correlation over the one-step objective; it closes part of the gap to a constant without reaching it.

`gaussian_smooth` also grew a `causal` flag — a centered kernel reads ~480 ms of future into every
feature, which flatters an offline baseline against a streaming decoder. Under the corrected
protocol the classical Wiener-filter reference is R² 0.31, down from 0.54 measured the old way.
Cross-session and NLB results are unaffected: neither touches minival.

## Architecture

```
spikes ─► tokenizer ─► encoder ─► z ─► world model ─► ẑ(t+1)
          per-unit     transformer    action-cond.    JEPA + forecast + rollout
          embeddings   or state-space  causal
                                  │
                                  ├─► Poisson rate head ─► co-bps
                                  └─► velocity decoder  ─► kinematics
```

- **Tokenizer** — per-unit learned embeddings; permutation-invariant and cross-session, so a
  population of any channel count or layout maps to one token per bin.
- **Encoder** — a rotary temporal transformer or a bidirectional diagonal state-space model
  (S5/LRU-style). With a learnable per-mode timescale (`--ssm --ssm-dt`) the state-space encoder is
  the strongest single model (0.343 on validation).
- **World model** — an action-conditioned next-latent predictor (EMA-target JEPA + forecast +
  scheduled-sampling rollout loss for drift-resistant open-loop simulation).
- **Ensemble** — rate-space greedy selection over diverse members plus MINT, a non-neural
  trajectory library; selection and smoothing are tuned on a train-carved split.

## Quickstart

```bash
pip install -e ".[dev]" && pytest -q
scripts/nlb.sh mc_maze 000128                               # train a model
python -m noema.eval.score_test --submission submission.h5  # official test co-bps
```

Device-agnostic (CUDA · MPS · CPU). Ensemble regeneration and the member pool live in
[`scripts/nlb_submission.md`](scripts/nlb_submission.md). Validation and minival numbers elsewhere
in the repo are development figures, labeled as such.

## License

MIT — see [LICENSE](LICENSE).
