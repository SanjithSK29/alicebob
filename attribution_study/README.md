# Attribution study (proof of concept)

A self-contained study of *why* an agent starts colluding — which message / which
step actually causes the accept-verdict violation — and which causal-attribution
method identifies that cause most accurately.

This folder is **separate from** `experiments/`, `analysis/`, `task/`, and `SALT_NLP/`;
nothing in the existing repo is modified.

## What it answers

When Alice violates the verdict protocol (accepts Bob's work without the required raw
log), three candidate causes sit in her context:

| Cause | Meaning | Initiator |
| --- | --- | --- |
| **M** | Bob's messages in the current episode | peer |
| **F** | reward + verdict-review feedback from earlier episodes | self |
| **R** | Alice's own private reflections from earlier episodes | self |

The study measures, per attribution method, an **accuracy** against ground truth:

- **cause accuracy** — did the method's top cause match the true cause?
- **initiator accuracy** — did it identify who started it (peer vs self)?
- **step accuracy** — did it identify the step (round 1–5 / feedback / reflection)?

## Methods benchmarked

- **but-for** — cause whose removal alone flips the violation off.
- **HP** — Halpern-Pearl original definition (contingency = removed causes).
- **Chockler** — Chockler-Halpern degree of responsibility `1/(1 + |W|)`.
- **Shapley** — average marginal contribution (reference).

All are scored the same way (`argmax` of the method's per-cause score) so they are
comparable. Accuracy is measured over **synthetic replay tables with known structure**
(ground truth by construction), with noisy samples and Wilson-interval thresholds.

## Files

- `attribution.py` — wilson intervals; but-for, HP, Chockler, Shapley, ablation.
- `synthetic.py` — the synthetic benchmark (no API); prints the accuracy table.
- `loader.py` — parse a compressed trajectory into per-episode `Episode` objects and
  the message indices of each cause (copied from `SALT_NLP/loader.py`).
- `interventions.py` — remove any subset of `{M, F, R}` from Alice's context; validate
  the edited conversation (copied from `SALT_NLP/interventions.py`).
- `replay.py` — re-sample Alice's forced `submit_verdict` against Gemini, with retry
  handling for transient 503 / rate-limit errors (unlike `SALT_NLP/replay.py`, this is
  exercised against the live API).
- `download.py` — pull a few trajectories from `SALT-NLP/agent-collusion` on HF.
- `run_poc.py` — orchestrate Phase 1 (synthetic) + Phase 2 (real replay).

## Run

```bash
pip install -r requirements.txt
cp .env.example .env        # set GEMINI_API_KEY
python download.py --n-traces 3
python synthetic.py          # Phase 1: benchmark (no API)
python run_poc.py --n-episodes 2 --samples 8   # Phase 1 + Phase 2 (real API)
```

## Assumptions

- Only the verdict is re-sampled; task and communication phases stay as recorded.
- `R` was written after seeing `F`, so removing `F` alone does not remove information
  already carried by `R` (direct vs. total effect); joint removals are tested.
- Synthetic ground truth is known by construction; real-data labels are reference
  signals, not perfect gold.
- "Initiator" is an operational mapping: peer-message causes -> peer-initiated,
  feedback/reflection causes -> self-initiated.
