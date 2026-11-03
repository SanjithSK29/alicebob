"""Interventions on Alice's context before her verdict.

Candidate causes of a violation (accept) at episode e, as segments of Alice's context:
  M  the peer's messages delivered in episode e
  F  outcome feedback (reward + verdict review) of earlier episodes
  R  Alice's own private reflections (prompt + reply) of earlier episodes

An intervention *removes* a set of causes from the context (a do_context edit) and
leaves everything else untouched, so the replay measures the effect of that text
on the verdict. Design caveat (state it in the report): R was written after seeing F,
so removing F alone does not remove the information already carried by R; F and R
are therefore separate causes whose joint removal is also tested.
"""
from __future__ import annotations

from itertools import combinations

CAUSES = ("M", "F", "R")


def replay_context(ep, removed: frozenset[str] | set[str] = frozenset()) -> list[dict]:
    """Alice's context up to and including the verdict-phase prompt, minus the removed causes."""
    drop: set[int] = set()
    for c in removed:
        if c not in CAUSES:
            raise ValueError(f"unknown cause {c!r}")
        drop.update(ep.segments[c])
    ctx = [m for i, m in enumerate(ep.context[: ep.verdict_phase_idx + 1]) if i not in drop]
    return ctx


def all_subsets(causes=CAUSES):
    """Every subset of causes to remove (2^n), smallest first."""
    out = []
    for r in range(len(causes) + 1):
        out.extend(frozenset(c) for c in combinations(causes, r))
    return out


def validate(ctx: list[dict]) -> list[str]:
    """Structural checks that a (possibly edited) context is still a legal conversation."""
    problems = []
    if not ctx or ctx[0].get("role") != "system":
        problems.append("does not start with the system prompt")
    if ctx and ctx[-1].get("role") != "user":
        problems.append("does not end with the verdict-phase user prompt")
    elif ctx and "verdict phase" not in str(ctx[-1].get("content", "")):
        problems.append("last message is not the verdict-phase prompt")
    pending = set()
    for i, m in enumerate(ctx):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            pending = {tc["id"] for tc in m["tool_calls"]}
        elif m.get("role") == "tool":
            tid = m.get("tool_call_id")
            if tid not in pending:
                problems.append(f"orphan tool message at {i}")
            pending.discard(tid)
    return problems
