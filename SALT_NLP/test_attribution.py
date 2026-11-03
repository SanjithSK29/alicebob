"""Unit tests: the methods must recover known answers on tables built from known causal structures."""
import random
from attribution import all_methods, hp, _subsets

C = ("M", "F", "R")
HI, LO = 0.9, 0.05  # violation probability when the structure says 'violates' / 'does not violate'

def make_table(structure, n=20, rng=None):
    t = {}
    for S in _subsets(C):
        present = set(C) - set(S)
        p = HI if structure(present) else LO
        k = round(p * n) if rng is None else sum(rng.random() < p for _ in range(n))
        t[frozenset(S)] = (k, n)
    return t

STRUCTURES = {
    "single (M only)":      lambda p: "M" in p,
    "OR (M or F)":          lambda p: "M" in p or "F" in p,
    "AND (M and F)":        lambda p: "M" in p and "F" in p,
    "3-way OR (M,F,R)":     lambda p: bool(p & {"M", "F", "R"}),
}

def show(name, res):
    row = lambda d: " ".join(f"{c}={d[c]:.2f}" for c in C)
    print(f"{name:18s} but_for[{row(res['but_for'])}]  ablation[{row(res['ablation'])}]  shapley[{row(res['shapley'])}]  resp[{row(res['responsibility'])}]")

if __name__ == "__main__":
    print("exact counts (n=20)")
    exp = {
        "single (M only)":  dict(but_for=dict(M=1,F=0,R=0), resp=dict(M=1,F=0,R=0)),
        "OR (M or F)":      dict(but_for=dict(M=0,F=0,R=0), resp=dict(M=.5,F=.5,R=0)),
        "AND (M and F)":    dict(but_for=dict(M=1,F=1,R=0), resp=dict(M=1,F=1,R=0)),
        "3-way OR (M,F,R)": dict(but_for=dict(M=0,F=0,R=0), resp=dict(M=1/3,F=1/3,R=1/3)),
    }
    for name, f in STRUCTURES.items():
        r = all_methods(make_table(f), C); show(name, r)
        for m, key in (("but_for", "but_for"), ("responsibility", "resp")):
            for c in C:
                assert abs(r[m][c] - exp[name][key][c]) < 1e-9, (name, m, c, r[m][c], exp[name][key][c])
    # Shapley cannot tell OR from AND; HP/responsibility can
    s_or = all_methods(make_table(STRUCTURES["OR (M or F)"]), C)["shapley"]
    s_and = all_methods(make_table(STRUCTURES["AND (M and F)"]), C)["shapley"]
    assert all(abs(s_or[c] - s_and[c]) < 1e-9 for c in C)
    print("\nassertions passed: but-for and responsibility match the hand-derived values; Shapley identical for OR and AND")

    print("\nnoise robustness: 500 simulated tables per structure, n samples per cell")
    for n in (10, 20, 40):
        for name, f in STRUCTURES.items():
            rng = random.Random(1); ok = 0
            for _ in range(500):
                r = all_methods(make_table(f, n, rng), C)["responsibility"]
                ok += all(abs(r[c] - exp[name]["resp"][c]) < 1e-9 for c in C)
            print(f"  n={n:2d} {name:18s} responsibility exactly right in {ok/5:.0f}% of runs")
