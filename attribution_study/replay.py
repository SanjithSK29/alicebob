"""Re-sample Alice's verdict under interventions on her context.

Only the verdict is re-sampled; the task and communication phases stay as recorded.
This is a self-contained sampler with retry handling for transient provider errors
(503 / rate-limit), unlike SALT_NLP/replay.py which was marked untested.
"""
from __future__ import annotations

import json
import math
import os
import time
import random
from typing import Callable

from dotenv import load_dotenv

from attribution import wilson

load_dotenv()  # reads attribution_study/.env (GEMINI_API_KEY)

Sampler = Callable[[list[dict]], "str | None"]

# Forced submit_verdict tool call (matches the recorded protocol schema).
SUBMIT_VERDICT = {
    "type": "function",
    "function": {
        "name": "submit_verdict",
        "description": "Submit your final verdict on Bob's task for this episode.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Your reasoning for this verdict."},
                "verdict": {
                    "type": "string",
                    "enum": ["accept", "reject"],
                    "description": "accept if the evidence Bob transmitted meets both verdict criteria, reject otherwise.",
                },
            },
            "required": ["reasoning", "verdict"],
        },
    },
}

# Transient error substrings to retry (Gemini returns HTTP 503 under load; the
# sandbox also drops connections intermittently).
_TRANSIENT_MARKERS = ("503", "high demand", "rate", "quota", "timeout", "overloaded",
                      "temporarily", "try again", "network", "unreachable",
                      "connection", "errno", "reset", "apiconnection",
                      "resolution", "dns", "name resolution")


def _is_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(m in text for m in _TRANSIENT_MARKERS)


class LiteLLMSampler:
    """Sample Alice's forced submit_verdict call against Gemini, with retries."""

    def __init__(self, model: str = "gemini/gemini-3.1-flash-lite",
                 temperature: float = 1.0, max_tokens: int = 4096,
                 retries: int = 12, base_wait: float = 4.0, max_wait: float = 45.0):
        import litellm
        self.litellm = litellm
        litellm.suppress_debug_info = True
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.base_wait = base_wait
        self.max_wait = max_wait

    def __call__(self, ctx: list[dict]) -> str | None:
        # Drop provider-internal and reasoning fields the recorded trace may carry.
        # DeepSeek rejects forced tool_choice while thinking is on, and stale
        # reasoning_content from a prior call, so we disable thinking for the verdict.
        _DROP = {"provider_specific_fields", "reasoning_content", "reasoning",
                 "reasoning_items", "thinking_blocks"}
        msgs = [{k: v for k, v in m.items() if k not in _DROP} for m in ctx]
        extra = {}
        if "deepseek" in self.model.lower():
            extra["thinking"] = {"type": "disabled"}
        for attempt in range(self.retries + 1):
            try:
                resp = self.litellm.completion(
                    model=self.model,
                    messages=msgs,
                    tools=[SUBMIT_VERDICT],
                    tool_choice={"type": "function", "function": {"name": "submit_verdict"}},
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **extra,
                )
                calls = resp.choices[0].message.tool_calls or []
                if not calls:
                    return None
                try:
                    verdict = json.loads(calls[0].function.arguments).get("verdict")
                except Exception:
                    return None
                if verdict in ("accept", "reject"):
                    return verdict
                return None
            except Exception as exc:
                if attempt == self.retries or not _is_transient(exc):
                    print(f"    [replay error, giving up] {type(exc).__name__}: {str(exc)[:160]}")
                    return None
                wait = min(self.base_wait * (2 ** attempt), self.max_wait) + random.uniform(0, 1.0)
                print(f"    [replay retry {attempt + 1}/{self.retries} in {wait:.1f}s] "
                      f"{type(exc).__name__}")
                time.sleep(wait)
        return None


def estimate(ctx: list[dict], sampler: Sampler, K: int) -> dict:
    """K independent verdict samples -> violation probability with Wilson interval."""
    out = [sampler(ctx) for _ in range(K)]
    valid = [o for o in out if o in ("accept", "reject")]
    k = sum(o == "accept" for o in valid)
    lo, hi = wilson(k, len(valid))
    p = (k / len(valid)) if valid else math.nan
    return {"p": p, "lo": lo, "hi": hi, "n": len(valid), "failed": K - len(valid)}


def estimate_parallel(ctx: list[dict], sampler: Sampler, K: int, workers: int = 6) -> dict:
    """Like estimate(), but sample concurrently to cut through provider throttling."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(lambda _: sampler(ctx), range(K)))
    valid = [o for o in out if o in ("accept", "reject")]
    k = sum(o == "accept" for o in valid)
    lo, hi = wilson(k, len(valid))
    p = (k / len(valid)) if valid else math.nan
    return {"p": p, "lo": lo, "hi": hi, "n": len(valid), "failed": K - len(valid)}


class DeepSeekFaithfulSampler:
    """Faithful replay for DeepSeek thinking models.

    The original run recorded DeepSeek with thinking on (reasoning_effort='high')
    and the verdict loop used tool_choice='auto' (forced tool_choice is rejected in
    thinking mode). We keep reasoning_content, enable thinking, use auto tool_choice,
    and retry missing tool calls -- mimicking the original verdict-phase loop.
    """

    def __init__(self, model: str = "deepseek/deepseek-v4-flash",
                 reasoning_effort: str = "high", temperature: float = 1.0,
                 max_tokens: int = 64000, retries: int = 6,
                 base_wait: float = 4.0, max_wait: float = 45.0):
        import litellm
        self.litellm = litellm
        litellm.suppress_debug_info = True
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.base_wait = base_wait
        self.max_wait = max_wait

    def __call__(self, ctx: list[dict]) -> str | None:
        # Keep reasoning_content (it is part of the recorded history); drop only
        # provider-internal plumbing.
        msgs = [{k: v for k, v in m.items() if k != "provider_specific_fields"} for m in ctx]
        for attempt in range(self.retries + 1):
            try:
                resp = self.litellm.completion(
                    model=self.model,
                    messages=msgs,
                    tools=[SUBMIT_VERDICT],
                    tool_choice="auto",
                    reasoning_effort=self.reasoning_effort,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                calls = resp.choices[0].message.tool_calls or []
                if calls:
                    try:
                        verdict = json.loads(calls[0].function.arguments).get("verdict")
                    except Exception:
                        verdict = None
                    if verdict in ("accept", "reject"):
                        return verdict
                # missing/malformed tool call: soft retry (mimics original loop)
                if attempt == self.retries:
                    return None
                wait = min(self.base_wait * (2 ** attempt), self.max_wait)
                time.sleep(wait + random.uniform(0, 1.0))
            except Exception as exc:
                if attempt == self.retries or not _is_transient(exc):
                    print(f"    [replay error, giving up] {type(exc).__name__}: {str(exc)[:160]}")
                    return None
                wait = min(self.base_wait * (2 ** attempt), self.max_wait)
                print(f"    [replay retry {attempt + 1}/{self.retries} in {wait:.1f}s] "
                      f"{type(exc).__name__}")
                time.sleep(wait + random.uniform(0, 1.0))
        return None
