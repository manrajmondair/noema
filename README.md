# Noema

A world model for neural population activity.

Most neural decoders treat the brain as a signal to classify: activity in, intent out. Noema takes the opposite stance — it models the population as a dynamical system and learns to predict its own future. Concretely, it fits

```
z_{t+1} ~ P(· | z_t, action_t, context_t)
```

over a learned latent state `z`. A forward model of this kind gives decoding, drift-robust calibration, a closed-loop simulator, and a per-subject digital twin as downstream consequences of a single objective.

## Design

- **Tokenizer** — every recorded unit owns a learned embedding, so a session's population vector becomes one token per time bin regardless of channel count or layout. Permutation-invariant and portable across sessions and subjects.
- **Encoder** — a rotary temporal transformer maps the token stream to latent states `z_t`.
- **World model** — an action-conditioned, causal predictor trained to forecast the next latent (joint-embedding style, against an EMA target encoder), and to roll out autoregressively.
- **Heads** — a Poisson readout for firing rates (co-smoothing) and a behavior decoder for kinematics.

## Layout

```
noema/
  models/   tokenizer, encoder, world_model, heads, assembly
  data/     dataset loaders + synthetic generator
  train/    training entrypoints
  sim/      closed-loop rollout environment
  eval/     benchmark harnesses (NLB, FALCON)
tests/      wiring + overfit checks
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Training runs are GPU-first but the code is device-agnostic (CUDA → MPS → CPU). Benchmark and dataset extras install with `pip install -e ".[data,train]"`.

## Roadmap

Neural Latents Benchmark (co-bps, velocity R²) → FALCON cross-session few-shot → closed-loop simulator → sensory coupling → interactive demo.
