"""Download a small slice of the SALT-NLP/agent-collusion dataset for the POC.

Loads the episodes table, filters to the controlled-peer Gemini pilot condition,
and downloads a few compressed trajectories into ./traces (gitignored).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

PILOT_CONDITION = "controlled_peer__gemini-3-1-flash-lite__summary--accept--full-feedback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traces", type=int, default=3)
    ap.add_argument("--condition", default=PILOT_CONDITION)
    args = ap.parse_args()

    ds = load_dataset("SALT-NLP/agent-collusion", "episodes", split="test")
    df = ds.to_pandas()
    pilot = df[df.condition_id == args.condition]
    print(f"pilot condition: {pilot.shape[0]} episodes, "
          f"alice accept rate = {pilot.alice_verdict.eq('accept').mean():.3f}")

    trace_files = pilot.trace_file.unique()[: args.n_traces]
    out = Path("traces")
    out.mkdir(exist_ok=True)
    for tf in trace_files:
        local = hf_hub_download("SALT-NLP/agent-collusion", tf, repo_type="dataset",
                                local_dir=str(out))
        print(f"  downloaded {local}")
    print(f"done: {len(trace_files)} traces in {out}/")


if __name__ == "__main__":
    main()
