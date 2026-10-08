"""Pools M3 runs that differ only in seed, from the per-task bit strings their summaries print.

Usage: python scripts/pool_m3.py results/kaggle-m3-seed0-*.txt results/kaggle-m3-seed1-*.txt ...

Every run is scored on the same tasks (same ids hash) and starts from the same model, so the
"before" bits must be identical. Tasks are the independent unit: each task's change is averaged
over seeds, and a paired sign-flip permutation test asks whether the mean change could be zero.
"""
import random
import sys


def load(path):
    bits = {}
    for line in open(path):
        p = line.split()
        if len(p) == 4 and p[0] == "tasks" and p[2] in ("before", "after"):
            bits.setdefault("ids", p[1])
            if p[1] != bits["ids"]:
                raise SystemExit(f"{path}: two task sets in one file")
            bits[p[2]] = [int(c) for c in p[3]]
    if "before" not in bits or "after" not in bits:
        raise SystemExit(f"{path}: no 'tasks <ids> before/after <bits>' lines")
    return bits


def sign_flip_p(diffs, rounds=200_000, seed=0):
    """Two-sided p for mean(diffs) != 0, flipping each task's sign at random (plus-one corrected)."""
    rng = random.Random(seed)
    nz = [d for d in diffs if d]
    obs = abs(sum(nz))
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in nz)) >= obs - 1e-12 for _ in range(rounds))
    return (hits + 1) / (rounds + 1)


def pool(runs):
    ids, before = runs[0]["ids"], runs[0]["before"]
    for r in runs:
        if r["ids"] != ids or r["before"] != before:
            raise SystemExit("runs disagree on the task set or the starting model's results")
    n, k = len(before), len(runs)
    after = [sum(r["after"][i] for r in runs) / k for i in range(n)]
    diffs = [a - b for a, b in zip(after, before)]
    fixed = [sum(1 for i in range(n) if not before[i] and r["after"][i]) for r in runs]
    broke = [sum(1 for i in range(n) if before[i] and not r["after"][i]) for r in runs]
    return {
        "n": n, "seeds": k, "ids": ids,
        "before": sum(before) / n,
        "after_each": [sum(r["after"]) / n for r in runs],
        "after_mean": sum(after) / n,
        "fixed_each": fixed, "broke_each": broke,
        "fixed_in_all": sum(1 for i in range(n) if not before[i] and all(r["after"][i] for r in runs)),
        "broke_in_all": sum(1 for i in range(n) if before[i] and not any(r["after"][i] for r in runs)),
        "p": sign_flip_p(diffs),
    }


if __name__ == "__main__":
    s = pool([load(p) for p in sys.argv[1:]])
    each = ", ".join(f"{100 * a:.1f}%" for a in s["after_each"])
    print(f"{s['seeds']} seeds, same {s['n']} tasks ({s['ids']}): pass@1 {100 * s['before']:.1f}% -> {each}; "
          f"mean {100 * s['after_mean']:.1f}% (+{100 * (s['after_mean'] - s['before']):.1f} points)")
    print(f"fixed per seed {s['fixed_each']}, broke per seed {s['broke_each']}; "
          f"fixed in every seed {s['fixed_in_all']}, broke in every seed {s['broke_in_all']}")
    print(f"paired sign-flip test over tasks (seeds averaged): p = {s['p']:.2g}")
