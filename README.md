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

| FALCON H1 held-in | FALCON H1 zero-shot | World-model rollout |
|---:|---:|---:|
| R² 0.87 | R² ~0.6 | corr 0.45 over 10 bins |

The world model predicts firing one step ahead at corr ~0.47 and, with a rollout objective, holds
near 0.45 across the horizon instead of decaying.

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
