"""M4: branching rollouts from mid-episode snapshots, against replaying from the start.

An episode is a shared prefix (set up a workspace and build the state the agent works in) followed
by D more steps. A rollout tree branches K ways after the prefix and after every step, so it has
K**D leaves. Two ways to produce all leaves:

  fork:   run the prefix once; at every node fork K children and run the next step in each;
  replay: for every leaf, start a new sandbox and run the prefix and the D steps on its path.

For each tree this prints the wall time, the number of steps executed, and checks:
  * every leaf ends in the same state under both strategies (a digest of the sandbox's state);
  * with a prefix that records something no replay can reproduce (a timestamp and a random id,
    as real environments do: log lines, temp names, server ports), how many leaves still see
    the episode's own prefix: all of them when forked, none when replayed.
Prints JSON lines.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from crucible.config import Limits  # noqa: E402
from crucible.sandbox import Zygote  # noqa: E402

# The prefix: a workspace of 200 source files and an index over a 300,000-entry table, the kind
# of state an agent builds by cloning a repository and running its setup. About 0.1-0.2 s.
PREFIX = (
    "import collections, hashlib, json, os, re, statistics\n"
    "os.makedirs('repo', exist_ok=True)\n"
    "for i in range(200):\n"
    "    open(f'repo/mod{i}.py', 'w').write(f'def f{i}(x):\\n    return x * {i}\\n' * 40)\n"
    "table = {i: hashlib.sha1(str(i).encode()).hexdigest() for i in range(300000)}\n"
    "index = collections.defaultdict(list)\n"
    "for k, v in table.items():\n"
    "    index[v[:3]].append(k)\n"
    "history = []\n"
)
# What a real environment also records during setup and no replay reproduces.
STAMP = "import time, uuid\nstamp = (time.time_ns(), uuid.uuid4().hex)\n"
# Step d on branch b: an edit and a check, about 10-20 ms.
STEP = (
    "history.append(({d}, {b}))\n"
    "open(f'repo/mod{b}.py', 'a').write('# edit {d}.{b}\\n')\n"
    "hits = sum(len(index[k]) for k in list(index)[:{n}])\n"
)
DIGEST = (
    "import hashlib\n"
    "h = hashlib.sha256(repr((history, hits)).encode())\n"
    "for name in sorted(os.listdir('repo')):\n"
    "    h.update(open('repo/' + name, 'rb').read())\n"
    "print(h.hexdigest())\n"
)


def step(d, b):
    return STEP.format(d=d, b=b, n=200 + 50 * d + b)


def run(sb, code):
    r = sb.exec(code, timeout=120)
    if r.get("error"):
        raise RuntimeError(r["error"])
    return r["stdout"]


def fork_tree(z, limits, k, depth, stamp):
    """All leaves by forking. Returns {path: (digest, stamp)}, steps run, seconds."""
    t = time.perf_counter()
    root = z.spawn(limits)
    run(root, PREFIX + (STAMP if stamp else ""))
    steps, leaves, level = 1, {}, [((), root)]
    for d in range(depth):
        nxt = []
        for path, sb in level:
            for b in range(k):
                child = sb.fork()
                run(child, step(d, b))
                steps += 1
                nxt.append((path + (b,), child))
            sb.close()
        level = nxt
    for path, sb in level:
        leaves[path] = (run(sb, DIGEST).strip(), run(sb, "print(stamp)").strip() if stamp else None)
        sb.close()
    return leaves, steps, time.perf_counter() - t


def replay_tree(z, limits, k, depth, stamp):
    t = time.perf_counter()
    steps, leaves = 0, {}
    paths = [()]
    for _ in range(depth):
        paths = [p + (b,) for p in paths for b in range(k)]
    for path in paths:
        with z.spawn(limits) as sb:
            run(sb, PREFIX + (STAMP if stamp else ""))
            steps += 1
            for d, b in enumerate(path):
                run(sb, step(d, b))
                steps += 1
            leaves[path] = (run(sb, DIGEST).strip(), run(sb, "print(stamp)").strip() if stamp else None)
    return leaves, steps, time.perf_counter() - t


def main():
    limits = Limits(memory_mb=4096, tmpfs_mb=256, processes=256)
    with Zygote(preload=["json", "hashlib", "collections"]) as z:
        print(json.dumps({"bench": "branch", "cpus": os.cpu_count(), "features": z.features}), flush=True)
        # the cost of the prefix alone, for scale
        t = time.perf_counter()
        with z.spawn(limits) as sb:
            run(sb, PREFIX)
        prefix_s = time.perf_counter() - t
        for k, depth in ((4, 1), (4, 2), (8, 2), (4, 3)):
            fl, fs, ft = fork_tree(z, limits, k, depth, stamp=False)
            rl, rs, rt = replay_tree(z, limits, k, depth, stamp=False)
            same = sum(fl[p][0] == rl[p][0] for p in fl)
            print(json.dumps({
                "k": k, "depth": depth, "leaves": len(fl), "prefix_s": round(prefix_s, 3),
                "fork": {"steps": fs, "s": round(ft, 2)}, "replay": {"steps": rs, "s": round(rt, 2)},
                "speedup": round(rt / ft, 1), "leaves_identical": f"{same}/{len(fl)}"}), flush=True)
        # a prefix that records a timestamp and a random id
        fl, _, _ = fork_tree(z, limits, 4, 2, stamp=True)
        rl, _, _ = replay_tree(z, limits, 4, 2, stamp=True)
        f_stamps = {v[1] for v in fl.values()}
        r_stamps = {v[1] for v in rl.values()}
        print(json.dumps({"nondeterministic_prefix": {
            "leaves": len(fl),
            "fork_distinct_prefix_states": len(f_stamps),
            "replay_distinct_prefix_states": len(r_stamps),
            "note": "forked leaves all share the one prefix the episode had; each replay rebuilt "
                    "a different one"}}), flush=True)


if __name__ == "__main__":
    main()
