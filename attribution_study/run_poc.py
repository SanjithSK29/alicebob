"""Proof-of-concept orchestrator: synthetic benchmark + real interventional replay.

Phase 1 (no API): synthetic known-structure accuracy benchmark.
Phase 2 (Gemini): replay Alice's verdict under every {M,F,R} removal on a few
real episodes, then score but-for / HP / Chockler responsibility.

Run:  python run_poc.py --n-episodes 2 --samples 8
"""
from __future__ import annotations

import argparse
import glob
import sys

import loader
from interventions import CAUSES, all_subsets, replay_context, validate
from attribution import all_methods, hp, but_for, responsibility
from replay import LiteLLMSampler, estimate

import synthetic


def real_replay_table(ep, sampler, K):
    """Build {removed_set: (k, n)} by re-sampling Alice's verdict K times per cell."""
    table = {}
    for removed in all_subsets():
        ctx = replay_context(ep, removed)
        problems = validate(ctx)
        if problems:
            print(f"    [validate FAILED for {set(removed)}] {problems}")
            sys.exit(1)
        est = estimate(ctx, sampler, K)
        k = round(est["p"] * est["n"]) if est["n"] else 0
        table[frozenset(removed)] = (k, est["n"])
        print(f"    removed={set(removed) or '{}'} -> p={est['p']:.3f} "
              f"[{est['lo']:.3f},{est['hi']:.3f}] n={est['n']} failed={est['failed']}")
    return table


def report_methods(table):
    res = all_methods(table, CAUSES)
    hp_res = hp(table, CAUSES)
    print("    causes: M = peer messages (current episode), F = feedback history, "
          "R = reflection history")
    print("    but-for (cause if removing it alone flips violation off): "
          f"{ {k: int(v) for k, v in res['but_for'].items()} }")
    print("    Halpern-Pearl (is_cause, min contingency |W|): "
          f"{ {k: (ok, w) for k, (ok, w, _) in hp_res.items()} }")
    print("    Chockler-Halpern responsibility 1/(1+|W|): "
          f"{ {k: round(v, 3) for k, v in res['responsibility'].items()} }")
    print("    Shapley: "
          f"{ {k: round(v, 3) for k, v in res['shapley'].items()} }")
    print("    ablation (p_drop): "
          f"{ {k: round(v, 3) for k, v in res['ablation'].items()} }")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=2)
    ap.add_argument("--samples", type=int, default=8, help="K verdict samples per cell")
    ap.add_argument("--model", default="gemini/gemini-3.1-flash-lite")
    ap.add_argument("--no-replay", action="store_true", help="skip Phase 2 (real API)")
    args = ap.parse_args()

    # ---- Phase 1: synthetic benchmark (no API) ----
    synthetic.run_benchmark()

    if args.no_replay:
        return

    # ---- Phase 2: real replay ----
    print("=" * 72)
    print(f"REAL INTERVENTIONAL REPLAY  ({args.model}, K={args.samples}/cell)")
    print("=" * 72)
    sampler = LiteLLMSampler(model=args.model)

    paths = sorted(glob.glob("traces/**/*.json.gz", recursive=True))
    if not paths:
        print("no traces under traces/ -- run `python download.py` first")
        sys.exit(1)

    episodes = []
    for p in paths:
        for ep in loader.load_trace(p):
            episodes.append((p, ep))

    # Prefer violation episodes (Alice accepted), then fall back to any episode.
    violations = [(p, e) for p, e in episodes if e.violation]
    chosen = violations[: args.n_episodes]
    if len(chosen) < args.n_episodes:
        chosen += [e for e in episodes if e not in chosen][: args.n_episodes - len(chosen)]

    for p, ep in chosen:
        print(f"\n-- {p.split('/')[-1]} episode {ep.episode_index}: "
              f"recorded verdict={ep.alice_verdict} --")
        table = real_replay_table(ep, sampler, args.samples)
        report_methods(table)

    print("\nPOC complete.")


if __name__ == "__main__":
    main()
