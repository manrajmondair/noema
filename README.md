# Noema

A world model of neural population dynamics. Rather than mapping activity to intent, it models the
population as a dynamical system and predicts its own future, `z_{t+1} ~ P(· | z_t, action_t)`, over
a learned latent state — so that decoding, calibration and simulation fall out of one objective.

**Half of that held.** The decoding half is competitive on a public benchmark. The forward-prediction
half was tested against trivial baselines and does not stand. Both are below; the retraction is the
more useful result.

## NLB MC_Maze — co-smoothing bits/spike, test split

| Method | co-bps | | Method | co-bps |
|---|---:|---|---|---:|
| GRAFT (ensemble) | 0.387 | | **Noema** | **0.367** |
| STNDT (ensemble) | 0.386 | | AutoLFADS | 0.336 |
| S5 | 0.382 | | MINT | 0.330 |
| STNDT (single) | 0.369 | | NDT | 0.323 |

Scored with `nlb_tools` against the public test labels — the leaderboard's metric and split.

This is an ensemble of 21 members plus MINT; the best single model is 0.333, so both "single" rows
above beat the whole ensemble with one model. The remaining gap is information-bound, not
architectural: 0.367 is 94% of the 0.392 known-condition oracle, and neither reweighting members,
nor a single model 0.010 stronger, nor a dedicated condition classifier moves it. What is left is
`H(condition | held-in)` — see [`scripts/nlb_submission.md`](scripts/nlb_submission.md).

## FALCON H1 — velocity decoding

| Split | R² |
|---|---:|
| Held-in (minival), window 50 / 75 | 0.545 / 0.603 |
| Cross-session, zero-shot | ~0.6 |

Three seeds each, official evaluator. Held-in is a **development split** and not
leaderboard-comparable: every `held-in-minival` recording is a byte-identical prefix of its
`held-in-calib` counterpart, so a run that fits on calibration and scores minival scores its own
training data. `disjoint_calib` excises the overlap; an earlier 0.87 was measured without it. The
configuration sweep also selected on this split. Cross-session is session-disjoint and shares no
non-zero row; nine adaptation levers in `eval/falcon_transfer.py` fail to close the drift gap.

## Forward prediction — does not hold

Rolled open-loop from a seed window, against a strictly causal flat forecast (the seed window's
mean, held constant across the horizon):

| | model | flat forecast |
|---|---:|---:|
| Per-neuron correlation across the recording | 0.195 – 0.264 | 0.232 – 0.272 |
| Population rate correlation | 0.374 – 0.527 | 0.452 – 0.511 |
| Forward bits/spike | **−0.17** | 0 by construction |

On correlation the rollout is level with a forecast containing no dynamics — ahead at the first
step, behind by the tenth, within ±0.04 throughout. `fp-bps` decides it: its null *is* the
per-neuron mean rate, so a constant scores exactly zero and the rollout is worse than knowing
nothing but each neuron's average firing.

An earlier 0.45 is retired rather than corrected. It was raw cross-neuron correlation, whose floor
is a property of the recording — for Poisson counts with rates λ a flat forecast scores
√(Var(λ)/(Var(λ)+E[λ])), 0.47 here and 0.02–0.93 across datasets. Reading the multi-step objective
as "drift-resistant, flat across the horizon" was that collapse, not a strength. Contamination was
not the cause: training on the overlap moves these by less than seed noise. The multi-step objective
is a real relative effect — it is why the model reaches parity with a constant rather than losing.

## Install and reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && pytest -q

scripts/nlb.sh mc_maze 000128                                # download and train one model
python -m noema.eval.score_test --submission submission.h5   # co-bps vs public test labels
scripts/falcon_data.sh                                       # FALCON H1 (~98 MB)
python -m noema.eval.falcon_run --data data/000954           # held-in velocity R²
python -m noema.eval.falcon_worldmodel --fidelity            # rollout vs the flat floor
```

Device-agnostic (CUDA, MPS, CPU). Run `scripts/gpu_smoke.py` on new hardware first: it checks fp32
precision, state-space kernel agreement against a sequential reference, and attention head
dimensions.

## Architecture

```
spikes ─▶ tokenizer ─▶ encoder ─▶ z_t ─▶ world model ─▶ ẑ_{t+1}
          per-unit     temporal /       action-cond.    JEPA + forecast
          embeddings   state-space      causal          + rollout loss
                                    │
                                    ├─▶ Poisson rate head ─▶ co-bps
                                    └─▶ velocity decoder  ─▶ kinematics
```

One learned embedding per unit makes any channel count or layout a single token per bin,
permutation-invariant and portable across sessions. The encoder is a rotary temporal transformer or
a bidirectional diagonal state-space model (S5/LRU) — the latter is the project's one architecture
win, 0.333 against 0.273. The ensemble is rate-space greedy selection over diverse members plus
MINT, a non-neural trajectory library, tuned on a train-carved split with the reported split scored
once.

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
