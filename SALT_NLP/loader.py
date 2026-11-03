"""Loader for the SALT-NLP agent-collusion traces (controlled-peer conditions).

Turns one trace (run.json.gz) into per-episode objects with the *candidate causes*
that Alice could have been influenced by when she submitted her verdict.

Segmentation of Alice's context follows the message layout of the released traces:
  user  "## Episode N: task phase"           -> start of episode N
  user  "[round=k/5][from=bob][type=...]"    -> a peer message delivered to Alice
  user  "## Episode N: verdict phase"        -> verdict phase begins
  user  "Episode N complete. ... ## Reward ... ## Verdict review ..."  -> outcome feedback
  user  "Reflect briefly ..." + assistant    -> private reflection
"""
from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

EP_START = re.compile(r"^## Episode (\d+): task phase")
EP_COMPLETE = re.compile(r"^Episode (\d+) complete\.")
PEER_MSG = re.compile(r"^\[round=(\d+)/(\d+)\]\[from=bob\]\[type=([^\]]*)\]\n\n(.*)", re.S)
REFLECT_PROMPT = re.compile(r"^Reflect briefly")
VERDICT_PHASE = re.compile(r"^## Episode (\d+): verdict phase")


@dataclass
class PastEpisode:
    """What Alice carries in memory about an earlier episode."""
    episode_index: int
    peer_messages: list[str]
    feedback: str          # reward + verdict-review text shown after the episode
    reflection: str        # Alice's own private reflection
    verdict: str | None


@dataclass
class Episode:
    run_id: str
    episode_index: int           # zero-based
    task_type: str
    alice_task_id: str
    bob_task_id: str
    alice_verdict: str           # 'accept' | 'reject'
    violation: bool              # Alice accepted although the protocol-compliant verdict is reject
    alice_reward: float
    alice_verdict_correct: bool | None
    peer_messages: list[str]     # Bob's messages delivered to Alice this episode (candidate cause 1)
    history: list[PastEpisode]   # earlier episodes: feedback (cause 2) and reflections (cause 3)
    context_prefix_len: int      # number of messages in Alice's context before this episode starts
    context: list[dict] = field(repr=False, default_factory=list)  # Alice's context up to this episode's verdict
    verdict_phase_idx: int = -1  # index of the user message that opens the verdict phase
    segments: dict = field(repr=False, default_factory=dict)  # cause name -> message indices in `context`


def _text(m: dict) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c)


def _verdict_from_tool(m: dict) -> str | None:
    try:
        d = json.loads(_text(m))
    except Exception:
        return None
    return d.get("verdict") if isinstance(d, dict) and d.get("recorded") else None


def parse_context(messages: list[dict]):
    """Split a cumulative context into per-episode blocks.

    Returns list of dicts keyed by episode number (1-based, as printed in the prompts).
    """
    blocks: dict[int, dict] = {}
    cur = None
    for i, m in enumerate(messages):
        role, text = m.get("role"), _text(m) if m.get("role") != "assistant" else (m.get("content") or "")
        if role == "user":
            s = EP_START.match(text)
            if s:
                cur = int(s.group(1))
                blocks[cur] = dict(start=i, peer_messages=[], peer_idx=[], reflect_idxs=[], feedback="", reflection="", verdict=None, verdict_idx=None)
                continue
            if cur is None:
                continue
            p = PEER_MSG.match(text)
            if p:
                blocks[cur]["peer_messages"].append(p.group(4))
                blocks[cur]["peer_idx"].append(i)
            elif VERDICT_PHASE.match(text):
                blocks[cur]["verdict_phase_idx"] = i
            elif EP_COMPLETE.match(text):
                blocks[cur]["feedback"] = text
                blocks[cur]["feedback_idx"] = i
            elif REFLECT_PROMPT.match(text):
                blocks[cur]["reflect_prompt_idx"] = i
                blocks[cur]["reflect_idxs"].append(i)
        elif role == "tool" and cur is not None and blocks[cur].get("verdict_phase_idx") is not None:
            v = _verdict_from_tool(m)
            if v:
                blocks[cur]["verdict"] = v
                blocks[cur]["verdict_idx"] = i
        elif role == "assistant" and cur is not None and blocks[cur].get("reflect_prompt_idx") is not None \
                and not blocks[cur]["reflection"] and isinstance(m.get("content"), str) and m["content"]:
            blocks[cur]["reflection"] = m["content"]
            blocks[cur]["reflect_idxs"].append(i)
    return blocks


def _segments(blocks: dict, k: int) -> dict:
    """Message indices of each candidate cause for a verdict in (1-based) episode k.

    M: the peer's messages in episode k.
    F: outcome feedback (reward + verdict review) of all earlier episodes.
    R: Alice's private reflections (prompt and reply) of all earlier episodes.
    """
    cur = blocks[k]
    return dict(
        M=sorted(cur["peer_idx"]),
        F=sorted(blocks[j]["feedback_idx"] for j in range(1, k) if "feedback_idx" in blocks[j]),
        R=sorted(i for j in range(1, k) for i in blocks[j].get("reflect_idxs", [])),
    )


def load_trace(path: str | Path) -> list[Episode]:
    d = json.load(gzip.open(path))
    run_id = d["run_id"]
    results = d["results"]
    # The last episode's context contains the whole run; earlier episodes' contexts are prefixes.
    full = results[-1]["agents"]["alice"]["messages"]
    blocks = parse_context(full)
    episodes: list[Episode] = []
    for r in results:
        k = r["episode_index"] + 1
        b = blocks[k]
        a = r["agents"]["alice"]
        verdict = a["verdict"]
        history = [
            PastEpisode(j - 1, blocks[j]["peer_messages"], blocks[j]["feedback"], blocks[j]["reflection"], blocks[j]["verdict"])
            for j in range(1, k)
        ]
        episodes.append(
            Episode(
                run_id=run_id,
                episode_index=r["episode_index"],
                task_type=r["task_type"],
                alice_task_id=a["task_id"],
                bob_task_id=r["agents"]["bob"]["task_id"],
                alice_verdict=verdict,
                violation=(verdict == "accept"),
                alice_reward=r["reward_by_agent"]["alice"],
                alice_verdict_correct=a.get("verdict_correct"),
                peer_messages=b["peer_messages"],
                history=history,
                context_prefix_len=b["start"],
                context=full[: (b["verdict_idx"] + 1) if b["verdict_idx"] is not None else len(full)],
                verdict_phase_idx=b["verdict_phase_idx"],
                segments=_segments(blocks, k),
            )
        )
    return episodes
