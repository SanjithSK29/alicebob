# Capstone Phase 2: attribution on the SALT-NLP collusion data

Question: when Alice (an LLM agent) violates the verification protocol by accepting Bob's work without the raw log, which of the things in her context caused it, and which blame method gives a sensible answer when several causes could each be enough?

## Data
Pilot slice of SALT-NLP/agent-collusion (Hugging Face): condition `controlled_peer__gemini-3-1-flash-lite__summary--accept--full-feedback`. Bob is scripted (summaries, always accept); Alice is gemini-3.1-flash-lite. 500 episodes (50 runs x 10); Alice accepts in 41.2%, matching the paper. `get_pilot.py` downloads the episodes table and 5 traces.

## Candidate causes (parts of Alice's context at her verdict)
- M: the peer's messages in the current episode
- F: reward + verdict-review feedback from earlier episodes
- R: Alice's own private reflections from earlier episodes

Note: the "agents" in the blame sense are information sources acting on Alice (peer, reward/feedback channel, her own history). The dataset has only two agents, so this is attribution among influence channels in a two-agent system, not among many agents.

## Files
- `loader.py`: trace -> Episode objects (verdict, peer messages, history, message indices of each cause).
- `interventions.py`: remove any subset of {M, F, R} from Alice's context; `validate` checks the edited conversation is still legal.
- `replay.py`: re-sample Alice's verdict K times -> P(violation) with a Wilson interval. `LiteLLMSampler` is NOT yet tested against a live API.
- `attribution.py`: but-for, ablation, Shapley, Halpern-Pearl (original definition) and Chockler-Halpern responsibility 1/(1+|W|) from a table {removed set -> (k, n)}. Thresholds hi=0.5, lo=0.25 are parameters to report.
- `test_attribution.py`: synthetic tables with known structure (single, OR, AND, 3-way OR). A test of the code, not a result.

## Status
Tested offline on 2 of the 5 traces (20 episodes): loader matches the episodes table; all 160 edited contexts validate. Nothing has been run against a live model yet (needs an API key). With clean effects (0.90 vs 0.05) about 40 samples per cell are needed for responsibility to be right ~91-97% of the time; 20 gives ~50-70%.

## Run
```
python -m pip install -U datasets pyarrow huggingface_hub pandas litellm
python get_pilot.py                      # downloads the pilot episodes table + 5 traces (not stored in this repo)
python test_attribution.py               # synthetic checks (Windows PowerShell: $env:PYTHONPATH="."; python test_attribution.py)
```
`get_pilot.py` writes `pilot_episodes.parquet` and `traces/` next to itself. Keep them out of git.

Next: fidelity check (re-sample with nothing removed, compare to recorded rate; needs a Gemini API key, read from an environment variable, never committed), then pilot replay.

## Assumptions to state in the report
- Only the verdict is re-sampled; task and communication phases stay as recorded.
- R was written after seeing F, so removing F alone does not remove its information (direct vs total effect).
- This is interventional replay with the factual history held fixed, not a full counterfactual.
