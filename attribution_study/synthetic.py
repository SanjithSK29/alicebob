"""Synthetic benchmark: how accurately each attribution method recovers a known
cause, initiator, and step from noisy replay tables.

Ground truth is known by construction, so this yields clean, statistically powered
accuracy numbers with zero API cost. It is the "what performs better" benchmark.
"""
from __future__ import annotations

import random

from attribution import (_subsets, but_for, hp, responsibility, shapley, ablation)

# Cause set: five communication rounds (peer-initiated) + feedback + reflection (self).
CAUSES = ("r1", "r2", "r3", "r4", "r5", "F", "R")


def initiator_of(cause: str) -> str:
    return "peer" if cause.startswith("r") else "self"


def step_label(cause: str) -> str:
    if cause.startswith("r"):
        return "round " + cause[1:]
    return {"F": "feedback", "R": "reflection"}[cause]


def make_table(structure, causes=CAUSES, n=20, hi=0.9, lo=0.05, rng=None):
    """structure(present:set)->bool. Returns {frozenset(removed): (k, n)}."""
    t = {}
    for S in _subsets(causes):
        present = set(causes) - set(S)
        p = hi if structure(present) else lo
        k = round(p * n) if rng is None else sum(rng.random() < p for _ in range(n))
        t[frozenset(S)] = (k, n)
    return t


# --- method -> scalar-score mapping, so every method has a comparable argmax ---

def _scores(name, result):
    if name == "hp":
        return {c: (1.0 if ok else 0.0) for c, (ok, _k, _w) in result.items()}
    return dict(result)  # but_for / responsibility / shapley / ablation already scalar


METHODS = ("but_for", "hp", "responsibility", "shapley")


def run_method(name, table, causes=CAUSES, hi=0.5, lo=0.25):
    if name == "but_for":
        return but_for(table, causes, hi, lo)
    if name == "hp":
        return hp(table, causes, hi, lo)
    if name == "responsibility":
        return responsibility(table, causes, hi, lo)
    if name == "shapley":
        return shapley(table, causes)
    raise ValueError(name)


def argmax(d):
    best_k, best_v = None, None
    for k, v in d.items():
        if best_v is None or v > best_v:
            best_k, best_v = k, v
    return best_k


# --- benchmark drivers ---

def _single_cause(cstar):
    return lambda present: cstar in present


def _or_same_initiator(cstar, d):
    assert initiator_of(cstar) == initiator_of(d)
    return lambda present: (cstar in present) or (d in present)


def _and(cstar, d):
    return lambda present: (cstar in present) and (d in present)


def single_cause_accuracy(n_sims=1000, n_per_cell=20, rng=None):
    """Accuracy of argmax==true-cause over single-cause tables (unambiguous initiator/step)."""
    rng = rng or random.Random(1)
    acc = {m: {"cause": 0, "initiator": 0, "step": 0} for m in METHODS}
    for _ in range(n_sims):
        cstar = rng.choice(CAUSES)
        table = make_table(_single_cause(cstar), n=n_per_cell, rng=rng)
        for m in METHODS:
            scores = _scores(m, run_method(m, table))
            pred = argmax(scores)
            acc[m]["cause"] += pred == cstar
            acc[m]["initiator"] += initiator_of(pred) == initiator_of(cstar)
            acc[m]["step"] += step_label(pred) == step_label(cstar)
    for m in METHODS:
        acc[m] = {k: v / n_sims for k, v in acc[m].items()}
    return acc


def or_same_accuracy(n_sims=1000, n_per_cell=20, rng=None):
    """On two-way OR with a shared initiator, does each method identify a responsible cause?
    but-for fails here (removing either cause alone never eliminates the violation)."""
    rng = rng or random.Random(2)
    pools = {"peer": [c for c in CAUSES if initiator_of(c) == "peer"],
             "self": [c for c in CAUSES if initiator_of(c) == "self"]}
    acc = {m: {"cause": 0, "initiator": 0} for m in METHODS}
    for _ in range(n_sims):
        grp = rng.choice(["peer", "self"])
        cstar, d = rng.sample(pools[grp], 2)
        table = make_table(_or_same_initiator(cstar, d), n=n_per_cell, rng=rng)
        for m in METHODS:
            scores = _scores(m, run_method(m, table))
            pred = argmax(scores)
            acc[m]["cause"] += pred in (cstar, d)
            acc[m]["initiator"] += initiator_of(pred) == grp
    for m in METHODS:
        acc[m] = {k: v / n_sims for k, v in acc[m].items()}
    return acc


def and_accuracy(n_sims=1000, n_per_cell=20, rng=None):
    """Two-way AND: both causes are necessary, so removing either alone flips the verdict."""
    rng = rng or random.Random(3)
    acc = {m: {"cause": 0} for m in METHODS}
    for _ in range(n_sims):
        cstar, d = rng.sample(CAUSES, 2)
        table = make_table(_and(cstar, d), n=n_per_cell, rng=rng)
        for m in METHODS:
            scores = _scores(m, run_method(m, table))
            pred = argmax(scores)
            acc[m]["cause"] += pred in (cstar, d)
    for m in METHODS:
        acc[m]["cause"] /= n_sims
    return acc


def run_benchmark():
    print("=" * 72)
    print("SYNTHETIC BENCHMARK  (ground truth known by construction)")
    print("=" * 72)
    for n in (10, 20, 40):
        acc = single_cause_accuracy(n_sims=1000, n_per_cell=n)
        print(f"\n-- single-cause recovery, n/cell = {n} --")
        print(f"{'method':<14} {'cause':>7} {'initiator':>11} {'step':>8}")
        for m in METHODS:
            print(f"{m:<14} {acc[m]['cause']:>7.3f} {acc[m]['initiator']:>11.3f} "
                  f"{acc[m]['step']:>8.3f}")

    print("\n-- two-way OR (shared initiator) at n/cell = 20 --")
    acc = or_same_accuracy()
    print(f"{'method':<14} {'cause':>7} {'initiator':>11}")
    for m in METHODS:
        print(f"{m:<14} {acc[m]['cause']:>7.3f} {acc[m]['initiator']:>11.3f}")

    print("\n-- two-way AND at n/cell = 20 --")
    acc = and_accuracy()
    print(f"{'method':<14} {'cause':>7}")
    for m in METHODS:
        print(f"{m:<14} {acc[m]['cause']:>7.3f}")
    print()


if __name__ == "__main__":
    run_benchmark()
