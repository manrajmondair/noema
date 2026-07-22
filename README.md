# Noema

**A world model of neural population dynamics.**

Most neural decoders treat the brain as a signal to classify: activity in, intent out. Noema
instead models the population as a dynamical system and learns to predict its own future,

```
z_{t+1} ~ P(· | z_t, action_t, context_t)
```

over a learned latent state `z`. The aim is to make decoding, few-shot calibration, a closed-loop
simulator, and a per-subject digital twin consequences of one objective rather than separate systems.

## Results

The co-bps and FALCON figures come from the datasets' official scorers on real recordings. co-bps
is measured against the public NLB test labels with `nlb_tools`, the same metric and split as the
leaderboard; the world-model rollout below is firing correlation on real held-out data.

**NLB MC_Maze, co-smoothing bits/spike (test split)**

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

Noema clears the long-standing strong baselines (AutoLFADS, MINT, NDT) and lands within 0.02
co-bps of the best ensembles. The gap to the top is information-bound rather than architectural.
0.367 is 94% of the 0.392 known-condition oracle, the score from handing each trial its
true-condition PSTH, which no published method exceeds. Three measurements pin the residual to
intrinsic ambiguity rather than model capacity: reweighting the ensemble members gains nothing
(they are already independent noise around a shared estimate), a single model 0.010 stronger moves
the ensemble by 0.0002, and a dedicated condition classifier is no sharper than the existing
consensus. What is left is `H(condition | held-in)`, the trials whose held-in neurons genuinely do
not determine the reach. The derivation and the full map of tested levers are in
[`scripts/nlb_submission.md`](scripts/nlb_submission.md).

**Forward model and transfer** (real recordings)

| Task | Metric | Result |
|---|---|---:|
| FALCON H1 velocity, held-in | R² (official evaluator) | 0.87 |
| FALCON H1 velocity, cross-session zero-shot | R² | ~0.6 |
| World-model rollout, real H1 | firing correlation over 10 bins | 0.45 |

The world model predicts population firing one step ahead at correlation ~0.47. Left alone its
open-loop accuracy decays over the horizon; a multi-step rollout objective holds it near 0.45
across all ten bins instead. Cross-session drift (0.87 to ~0.6) is a genuine wall; nine adaptation
levers are characterized in the eval code, none of them closing it.

## Architecture

```
 spikes ─▶ tokenizer ─▶ encoder ─▶  z_t  ─▶ world model ─▶ ẑ_{t+1}
           per-unit     temporal /          action-cond.   JEPA + forecast
           embeddings   state-space         causal         + rollout loss
                                       │
                                       ├─▶ Poisson rate head  → co-bps
                                       └─▶ velocity decoder    → kinematics
```

- **Tokenizer.** Every recorded unit owns a learned embedding, so a population becomes one token
  per time bin regardless of channel count or electrode layout. Permutation-invariant and portable
  across sessions and subjects, which is what makes cross-subject pretraining possible.
- **Encoder.** Pluggable: a rotary temporal transformer, or a bidirectional diagonal state-space
  model (S5/LRU-style). The state-space encoder with a learnable per-mode timescale
  (`--ssm --ssm-dt`) is the strongest single model, 0.343 on validation.
- **World model.** An action-conditioned causal predictor trained to forecast the next latent
  against an EMA target encoder, with a scheduled-sampling rollout loss for drift-resistant
  open-loop simulation.
- **Heads.** A Poisson rate readout for co-smoothing and a velocity decoder for kinematics.
- **Ensemble.** Rate-space greedy (Caruana) selection over diverse members plus MINT, a non-NN
  condition-averaged trajectory library. Selection and smoothing are tuned on a train-carved split;
  the reported split is left untouched.

## Layout

```
noema/
  models/   tokenizer, encoder, state-space encoder, world model, heads, session adversary
  data/     dataset, synthetic systems, multi-session pretraining
  train/    trainer, few-shot adaptation, CLI
  sim/      imagined rollouts
  eval/     metrics, baselines, NLB, MINT, ensemble, submission, score_test, FALCON, streaming
demo/       interactive in-browser world model, parity-checked against PyTorch
tests/      unit and integration checks
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && pytest -q

scripts/nlb.sh mc_maze 000128                             # download and train a single model
python -m noema.eval.score_test --submission submission.h5  # official co-bps vs public test labels
```

Training is GPU-first but device-agnostic (CUDA, MPS, CPU). Ensemble regeneration and the exact
member pool are documented in [`scripts/nlb_submission.md`](scripts/nlb_submission.md).

## Reporting

co-bps is the sequestered-test metric computed against the public labels; FALCON figures come from
the official evaluator on the sanctioned splits. Any validation or minival number elsewhere in the
repo is a development figure and is labeled as one.
