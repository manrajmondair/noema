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

**The world-model rollout figures are retired, and the forward-prediction claim with them.**
They were reported as raw cross-neuron correlation — correlating the predicted population vector
against the recorded one within a time bin. That estimator has a floor set by the recording rather
than by the model: for Poisson counts with per-channel rates λ, a flat forecast scores
√(Var(λ)/(Var(λ)+E[λ])), which is 0.47 on this data and ranges from 0.02 to 0.93 across datasets
with no model involved. It cannot be a headline, so it is no longer one.

Measured four ways, with the published objective restored (no behavior head) and a strictly causal
flat forecast — the seed window's mean, held constant across the horizon — scored identically:

| | model | flat forecast |
|---|---|---|
| cross-neuron within a bin, profile removed per session | 0.084 | **0.121** |
| per-neuron across the recording, variance-weighted | 0.232 | **0.475** |
| population rate correlation | 0.43 | **0.66** |
| forward bits/spike | **−0.17** | 0 by construction |

The flat forecast wins on all three correlation forms at every horizon. The fourth is the decisive
one: `fp-bps` takes the per-neuron mean rate as its null, so a constant scores exactly zero and a
negative value means the rollout is *worse than knowing nothing but each neuron's average firing
rate*. Ours is negative at every horizon, against 0.21–0.27 on the NLB leaderboard. There is no
baseline choice left to argue about.

Contamination was not the cause — training on the un-excised overlap moves these by less than seed
noise. The metric was. The multi-step objective is a real relative effect (0.232 against 0.053 for
one-step on the weighted form) and it closes part of the distance to a constant without reaching it.

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
