"""Bigger-scale attribution run: coarse {M,F,R} + round-level {r1..r5} attribution
over many real episodes, with concurrent sampling, resumable caching, and
cross-episode aggregation.

Run:  python run_big.py --episodes 6 --samples 30 --workers 8
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import loader
from interventions import CAUSES, ROUND_CAUSES, all_subsets, replay_context, validate
from attribution import all_methods, hp, wilson, but_for, responsibility, ablation
from replay import LiteLLMSampler, DeepSeekFaithfulSampler, estimate_parallel

CACHE_FILE = Path("cache_big.json")
RESULTS_FILE = Path("results_big.json")


# --------------------------------------------------------------------------- cache

def _cache_key(run_id, ep_index, removed) -> str:
    return f"{run_id}:{ep_index}:{'+'.join(sorted(removed)) or 'none'}"


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


# --------------------------------------------------------------------------- one cell

def sample_cell(ep, removed, causes, sampler, K, workers, cache, cache_key) -> dict:
    if cache_key in cache and cache[cache_key]["n"] >= K:
        return cache[cache_key]
    ctx = replay_context(ep, frozenset(removed), causes=causes)
    problems = validate(ctx)
    if problems:
        print(f"  [validate FAILED {set(removed)}] {problems}", flush=True)
        sys.exit(1)
    est = estimate_parallel(ctx, sampler, K, workers)
    k = round(est["p"] * est["n"]) if est["n"] else 0
    cell = {"k": k, "n": est["n"], "failed": est["failed"]}
    cache[cache_key] = cell
    save_cache(cache)
    print(f"    removed={set(removed) or '{}'} -> p={est['p']:.3f} "
          f"n={est['n']} failed={est['failed']}", flush=True)
    return cell


# --------------------------------------------------------------------------- drivers

def run_episode(ep, sampler, K, workers, cache) -> dict:
    tag = f"{ep.run_id} ep{ep.episode_index} (verdict={ep.alice_verdict})"
    print(f"\n=== {tag} ===", flush=True)

    # coarse {M,F,R}: full subset enumeration
    coarse = {}
    for removed in all_subsets(CAUSES):
        ck = _cache_key(ep.run_id, ep.episode_index, removed)
        coarse[frozenset(removed)] = sample_cell(ep, removed, CAUSES, sampler, K,
                                                 workers, cache, ck)
    coarse_table = {s: (c["k"], c["n"]) for s, c in coarse.items()}
    if any(n == 0 for _, n in coarse_table.values()):
        # A cell produced no valid samples (e.g. insufficient balance) -> cannot
        # compute attribution reliably; record the failure instead of crashing.
        print("  [skip: a coarse cell had 0 valid samples]", flush=True)
        return {"run_id": ep.run_id, "episode_index": ep.episode_index,
                "alice_verdict": ep.alice_verdict, "failed": True,
                "coarse": None, "round_ablation": None}
    coarse_methods = all_methods(coarse_table, CAUSES)

    # round {r1..r5}: baseline + each single-round removal (ablation / but-for)
    round_cells = {}
    for removed in [frozenset()] + [frozenset({r}) for r in ROUND_CAUSES]:
        ck = _cache_key(ep.run_id, ep.episode_index, removed)
        round_cells[removed] = sample_cell(ep, removed, ROUND_CAUSES, sampler, K,
                                           workers, cache, ck)
    p0 = round_cells[frozenset()]["k"] / round_cells[frozenset()]["n"] if round_cells[frozenset()]["n"] else 0.0
    round_ablation = {r: p0 - (round_cells[frozenset({r})]["k"] / round_cells[frozenset({r})]["n"]
                               if round_cells[frozenset({r})]["n"] else 0.0)
                      for r in ROUND_CAUSES}

    return {
        "run_id": ep.run_id,
        "episode_index": ep.episode_index,
        "alice_verdict": ep.alice_verdict,
        "coarse": {
            "responsibility": {k: round(v, 3) for k, v in coarse_methods["responsibility"].items()},
            "but_for": {k: int(v) for k, v in coarse_methods["but_for"].items()},
            "ablation": {k: round(v, 3) for k, v in coarse_methods["ablation"].items()},
            "factual_p": coarse[frozenset()]["k"] / coarse[frozenset()]["n"] if coarse[frozenset()]["n"] else None,
        },
        "round_ablation": {k: round(v, 3) for k, v in round_ablation.items()},
    }


def aggregate(episodes_results: list[dict]) -> dict:
    # Drop episodes that failed to sample (e.g. insufficient balance).
    episodes_results = [r for r in episodes_results if r.get("coarse")]
    n = len(episodes_results)
    # initiator: peer (M) vs self (F/R)
    initiator_top = {"peer": 0, "self": 0, "none": 0}
    for r in episodes_results:
        resp = r["coarse"]["responsibility"]
        top = max(resp, key=resp.get)
        if resp[top] <= 0:
            initiator_top["none"] += 1
        elif top == "M":
            initiator_top["peer"] += 1
        else:
            initiator_top["self"] += 1

    # which round's message, when removed, most reduces the violation
    round_top = {r: 0 for r in ROUND_CAUSES}
    for r in episodes_results:
        ra = r["round_ablation"]
        top = max(ra, key=ra.get)
        if ra[top] > 0:
            round_top[top] += 1

    avg_resp = {c: sum(r["coarse"]["responsibility"][c] for r in episodes_results) / n
                for c in CAUSES}
    return {
        "n_episodes": n,
        "initiator_top": initiator_top,
        "round_top": round_top,
        "avg_responsibility": {k: round(v, 3) for k, v in avg_resp.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--samples", type=int, default=30, help="K samples per cell")
    ap.add_argument("--workers", type=int, default=3, help="concurrent API calls")
    ap.add_argument("--model", default="gemini/gemini-3.1-flash-lite")
    ap.add_argument("--traces-dir", default="traces",
                    help="directory of downloaded .json.gz traces")
    args = ap.parse_args()

    if "deepseek" in args.model.lower():
        sampler = DeepSeekFaithfulSampler(model=args.model)
    else:
        sampler = LiteLLMSampler(model=args.model)
    cache = load_cache()
    print(f"cache: {len(cache)} cells; workers={args.workers}; K={args.samples}; "
          f"model={args.model}")

    paths = sorted(glob.glob(f"{args.traces_dir}/**/*.json.gz", recursive=True))
    episodes = []
    for p in paths:
        for ep in loader.load_trace(p):
            episodes.append(ep)
    # Prefer violation episodes (Alice accepted); these are where attribution matters.
    violations = [e for e in episodes if e.violation]
    chosen = violations[: args.episodes]
    print(f"{len(episodes)} episodes total, {len(violations)} violations; "
          f"running {len(chosen)}")

    results = []
    for ep in chosen:
        results.append(run_episode(ep, sampler, args.samples, args.workers, cache))
        RESULTS_FILE.write_text(json.dumps({"episodes": results,
                                            "aggregate": aggregate(results)},
                                           indent=2))

    agg = aggregate(results)
    print("\n" + "=" * 72)
    print("AGGREGATE")
    print("=" * 72)
    print(f"episodes: {agg['n_episodes']}")
    print(f"top-cause initiator: peer={agg['initiator_top']['peer']} "
          f"self={agg['initiator_top']['self']} none={agg['initiator_top']['none']}")
    print(f"top-round ablation: {agg['round_top']}")
    print(f"avg responsibility: {agg['avg_responsibility']}")
    print(f"\nresults -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
