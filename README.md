# Noema

Neural population modelling for brain–computer interfaces. A permutation-invariant spike tokenizer
maps any electrode layout to one token per time bin, a temporal transformer or bidirectional
state-space encoder builds a latent population state, and rate and velocity readouts decode from it.

## NLB MC_Maze — co-smoothing bits/spike, test split

| Method | co-bps | | Method | co-bps |
|---|---:|---|---|---:|
| GRAFT (ensemble) | 0.387 | | **Noema** | **0.367** |
| STNDT (ensemble) | 0.386 | | AutoLFADS | 0.336 |
| S5 | 0.382 | | MINT | 0.330 |
| STNDT (single) | 0.369 | | NDT | 0.323 |

Scored with `nlb_tools` against the public test labels — the leaderboard's metric and split. A
21-member ensemble plus MINT; the strongest single model is 0.333.

**The remaining gap is information-bound, not architectural.** 0.367 is 94% of the 0.392
known-condition oracle — the score from handing each trial its true-condition PSTH, which no
published method exceeds. Three measurements pin the residual to intrinsic ambiguity rather than
capacity: reweighting ensemble members gains nothing, a single model 0.010 stronger moves the
ensemble by 0.0002, and a dedicated condition classifier is no sharper than the existing consensus.
What is left is `H(condition | held-in)`, trials whose observed neurons do not determine the reach.
Derivation and the full map of tested levers: [`scripts/nlb_submission.md`](scripts/nlb_submission.md).

## FALCON H1 — velocity decoding

| Split | R² |
|---|---:|
| Held-in minival (development), window 50 / 75 | 0.545 / 0.603 |
| Cross-session zero-shot, session-disjoint | ~0.6 |

Three seeds each, official evaluator. In this dandiset every `held-in-minival` recording is a
byte-identical prefix of its `held-in-calib` counterpart, so a run that fits on calibration and
scores minival scores its own training data; `disjoint_calib` excises the overlap. Cross-session
holds out whole recording days from the same subject and shares no non-zero row with training —
that number measures the electrode drift the benchmark exists to test. Nine adaptation levers are
characterised in `eval/falcon_transfer.py`.

## Install and reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && pytest -q

scripts/nlb.sh mc_maze 000128                                # download and train one model
python -m noema.eval.score_test --submission submission.h5   # co-bps vs public test labels
scripts/falcon_data.sh                                       # FALCON H1 (~98 MB)
python -m noema.eval.falcon_run --data data/000954           # held-in velocity R²
```

Device-agnostic (CUDA, MPS, CPU). Run `scripts/gpu_smoke.py` on new hardware first: it checks fp32
precision, state-space kernel agreement against a sequential reference, and attention head
dimensions.

## Architecture

```
spikes ─▶ tokenizer ─▶ encoder ─▶ z_t ─▶ world model ─▶ ẑ_{t+1}
          per-unit     temporal /       auxiliary       JEPA + forecast
          embeddings   state-space      objective       + rollout loss
                                    │
                                    ├─▶ Poisson rate head ─▶ co-bps
                                    └─▶ velocity decoder  ─▶ kinematics
```

- **Tokenizer** — one learned embedding per unit, so any channel count or electrode layout becomes
  a single token per bin. Permutation-invariant and portable across sessions and subjects, which is
  what makes cross-subject pretraining possible.
- **Encoder** — a rotary temporal transformer or a bidirectional diagonal state-space model
  (S5/LRU). The state-space encoder is the strongest single model, 0.333 against 0.273.
- **World model** — an action-conditioned causal predictor trained against an EMA target encoder.
  Used as an auxiliary objective on the latent; `eval/falcon_worldmodel.py` measures its open-loop
  rollout against causal flat-forecast baselines and forward bits/spike.
- **Ensemble** — rate-space greedy selection over diverse members plus MINT, a non-neural
  condition-averaged trajectory library. Tuned on a train-carved split; the reported split is
  scored once.

## Layout

```
noema/  models/  tokenizer, encoder, state-space encoder, world model, heads
        data/    dataset, synthetic systems, NLB loader
        train/   trainer, few-shot adaptation, CLI
        eval/    metrics, baselines, NLB, MINT, ensemble, submission, FALCON, stimulation
        sim.py   open-loop rollout
tests/           unit and integration checks
scripts/         data fetch, cluster jobs, submission notes
```

MIT — see [LICENSE](LICENSE).
