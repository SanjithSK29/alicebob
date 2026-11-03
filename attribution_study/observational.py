"""Observational attribution over the *entire* SALT-NLP dataset (no API).

Answers, from recorded data alone:
  1. Which agent *proposes* collusion (from the agreement judge's first-side quote).
  2. Which communication round the proposal happens in.
  3. Which evaluation episode collusion onsets at, and via which pathway.
  4. Collusion rate by model.
  5. Proposer by model (does the initiator depend on who the model is?).

Results are printed and also written to results_observational.md.
"""
from __future__ import annotations

import re
from collections import Counter

from datasets import load_dataset


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def match_message(quote: str, communication: list) -> dict | None:
    q = norm(quote)
    if not q:
        return None
    hits = [m for m in communication if q in norm(m.get("content", ""))]
    if not hits:
        return None
    if len({m["sender"] for m in hits}) != 1:
        return None
    m = hits[0]
    return {"sender": m["sender"], "round": int(m.get("round", 0))}


def main():
    episodes = load_dataset("SALT-NLP/agent-collusion", "episodes", split="test")
    agreement = load_dataset("SALT-NLP/agent-collusion", "agreement", split="test")
    runs = load_dataset("SALT-NLP/agent-collusion", "runs", split="test")

    edf = episodes.to_pandas()
    adf = agreement.to_pandas()
    rdf = runs.to_pandas()

    out = []
    def p(*args):
        line = " ".join(str(a) for a in args)
        print(line)
        out.append(line)

    p("=" * 72)
    p("OBSERVATIONAL ATTRIBUTION  (full dataset, no API)")
    p("=" * 72)

    # ---- 1. onset ----
    p("\n[1] Collusion onset pathway (from runs table)")
    for k, v in rdf.onset_pathway.value_counts(dropna=False).items():
        p(f"    {str(k):>12}: {v}")
    onset = rdf.onset_evaluation_episode.dropna()
    p(f"\n    onset episode distribution (n={len(onset)} runs with an onset):")
    for ep, cnt in sorted(onset.astype(int).value_counts().items()):
        p(f"      episode {ep}: {cnt}")

    # ---- proposer mapping ----
    pos = adf[(adf.status == "valid") & (adf.label == True)].copy()
    comm = {}
    for row in episodes:
        comm[(row["run_id"], row["episode_index"])] = row["communication"]
    model_by_run = dict(zip(rdf["run_id"], zip(rdf["alice_model"], rdf["bob_model"])))

    rows = []  # (experiment, sender, round, alice_model, bob_model)
    for _, r in pos.iterrows():
        key = (r["run_id"], r["episode_index"])
        c = comm.get(key)
        if c is None:
            continue
        m = match_message(r["first_side_quote"], c)
        if m:
            am, bm = model_by_run.get(r["run_id"], ("", ""))
            rows.append((r["experiment"], m["sender"], m["round"], am, bm))

    # ---- 2. proposer overall ----
    p(f"\n[2] Who proposes collusion? (matched {len(rows)} of {len(pos)} positive quotes)")
    sender_counts = Counter(s for _, s, _, _, _ in rows)
    tot = sum(sender_counts.values())
    for s in ("alice", "bob"):
        n = sender_counts.get(s, 0)
        p(f"    {s:>6}: {n}  ({n / tot * 100:.1f}%)")

    p("\n    by experiment:")
    by_exp = {}
    for exp, s, _, _, _ in rows:
        by_exp.setdefault(exp, Counter())[s] += 1
    for exp in sorted(by_exp):
        c = by_exp[exp]
        n = sum(c.values())
        a, b = c.get("alice", 0), c.get("bob", 0)
        p(f"      {exp:<18} alice={a:>4} bob={b:>4}  (alice {a / n * 100:.0f}%)")

    # ---- 3. round ----
    p("\n[3] Which communication round carries the proposal? (round 1 = first)")
    round_counts = Counter(rd + 1 for _, _, rd, _, _ in rows)
    for rnd in range(1, 6):
        n = round_counts.get(rnd, 0)
        p(f"    round {rnd}: {n}  ({n / len(rows) * 100:.1f}%)")

    # ---- 4. collusion rate by model ----
    p("\n[4] Collusion rate by model (same-model 'main' experiment, eval episodes)")
    main = edf[(edf.experiment == "main") & (edf.is_warmup == False)]
    for model in sorted(main.alice_model.unique()):
        sub = main[main.alice_model == model]
        p(f"    {model:<28} collusion={sub.collusion.mean():.3f}  (n={len(sub)})")

    # ---- 5. proposer by model ----
    p("\n[5] Proposer by model (fraction of episodes where ALICE proposes)")
    def tabulate(model_field):
        by_model = {}
        for exp, s, _, am, bm in rows:
            m = am if model_field == "alice" else bm
            if not m:
                continue
            by_model.setdefault(m, Counter())[s] += 1
        for m in sorted(by_model, key=lambda x: -sum(by_model[x].values())):
            c = by_model[m]
            n = sum(c.values())
            a = c.get("alice", 0)
            p(f"    {m:<40} alice-proposes={a / n * 100:5.1f}%  (n={n})")

    p("\n    as Alice (model playing the Alice role):")
    tabulate("alice")
    p("\n    as Bob (model playing the Bob role -> does a stronger Bob make Alice propose less?):")
    tabulate("bob")

    # ---- save ----
    with open("results_observational.md", "w") as f:
        f.write("# Observational attribution results (full SALT-NLP dataset)\n\n")
        f.write("\n".join(out) + "\n")
    p("\nSaved to results_observational.md")


if __name__ == "__main__":
    main()
