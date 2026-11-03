"""Blame methods over replay results.

Input: a table mapping each *removed set* S (frozenset of cause names) to the replay
outcome (k violations out of n samples) when exactly the causes in S are removed from
Alice's context. The empty set is the factual condition.

    table[frozenset()]        -> violations with nothing removed
    table[frozenset({"M"})]   -> violations with the peer's messages removed
    ...

Methods (all return {cause: score}):
  but_for        1 if removing the cause ALONE significantly flips the violation off
  ablation       drop in violation probability when the cause alone is removed
  shapley        average marginal drop over all orders of removing causes
                 (value of a coalition T of removed causes: v(T) = p(empty) - p(T))
  hp             Halpern-Pearl (original definition, contingencies set to 'removed'):
                 c is a cause if some other set W of causes can be removed such that
                 violation still occurs with W removed but disappears when c is removed too
  responsibility Chockler-Halpern degree of responsibility: 1 / (1 + min |W|)

Because outcomes are noisy, 'violation occurs' and 'violation disappears' are decided
with Wilson intervals and two thresholds (hi, lo), which are parameters to report.
"""
from __future__ import annotations

from itertools import combinations
from math import sqrt

Table = dict  # frozenset[str] -> (k, n)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def prob(table: Table, S) -> float:
    k, n = table[frozenset(S)]
    return k / n


def _subsets(items, include_empty=True):
    items = list(items)
    for r in range(0 if include_empty else 1, len(items) + 1):
        for c in combinations(items, r):
            yield frozenset(c)


def persists(table: Table, S, hi: float) -> bool:
    """Violation clearly still occurs with S removed: lower CI bound >= hi."""
    k, n = table[frozenset(S)]
    return wilson(k, n)[0] >= hi


def eliminated(table: Table, S, lo: float) -> bool:
    """Violation clearly gone with S removed: upper CI bound <= lo."""
    k, n = table[frozenset(S)]
    return wilson(k, n)[1] <= lo


def but_for(table: Table, causes, hi: float = 0.5, lo: float = 0.25) -> dict:
    base = frozenset()
    out = {}
    for c in causes:
        out[c] = 1.0 if (persists(table, base, hi) and eliminated(table, {c}, lo)) else 0.0
    return out


def ablation(table: Table, causes) -> dict:
    p0 = prob(table, frozenset())
    return {c: max(0.0, p0 - prob(table, {c})) for c in causes}


def shapley(table: Table, causes) -> dict:
    causes = list(causes)
    n = len(causes)
    p0 = prob(table, frozenset())

    def v(T):
        return p0 - prob(table, T)

    from math import factorial
    out = {}
    for c in causes:
        others = [x for x in causes if x != c]
        total = 0.0
        for T in _subsets(others):
            w = factorial(len(T)) * factorial(n - len(T) - 1) / factorial(n)
            total += w * (v(T | {c}) - v(T))
        out[c] = total
    return out


def hp(table: Table, causes, hi: float = 0.5, lo: float = 0.25) -> dict:
    """Return {cause: (is_cause, min_contingency_size or None, witness set or None)}.

    AC1: violation occurs in the factual run (checked on the empty removal).
    AC2: exists W (other causes) with violation persisting under W and gone under W + c.
    AC3: minimality is by taking the smallest such W (responsibility uses it).
    """
    causes = list(causes)
    out = {}
    factual = persists(table, frozenset(), hi)
    for c in causes:
        best = None
        if factual:
            others = [x for x in causes if x != c]
            for W in sorted(_subsets(others), key=len):
                if persists(table, W, hi) and eliminated(table, W | {c}, lo):
                    best = W
                    break
        out[c] = (best is not None, None if best is None else len(best), best)
    return out


def responsibility(table: Table, causes, hi: float = 0.5, lo: float = 0.25) -> dict:
    """Chockler-Halpern degree of responsibility: 1/(1+k), k = size of the smallest contingency."""
    return {c: (1.0 / (1 + k) if ok else 0.0) for c, (ok, k, _) in hp(table, causes, hi, lo).items()}


def all_methods(table: Table, causes, hi: float = 0.5, lo: float = 0.25) -> dict:
    return dict(
        but_for=but_for(table, causes, hi, lo),
        ablation=ablation(table, causes),
        shapley=shapley(table, causes),
        responsibility=responsibility(table, causes, hi, lo),
    )
