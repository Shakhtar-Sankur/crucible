"""M1: forking a live sandbox.

1. fork latency as the sandbox's Python state and workspace grow;
2. memory: N branches of a sandbox holding a large state, each measured from inside
   (proportional set size: shared pages divided among the processes sharing them);
3. branching vs replaying: a state that takes a while to build (imports + computation),
   reached by forking it vs by starting a new sandbox and building it again.
Prints JSON lines."""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from crucible.config import Limits  # noqa: E402
from crucible.sandbox import Zygote  # noqa: E402

PSS = ("pss = 0\n"
       "for line in open('/proc/self/smaps_rollup'):\n"
       "    if line.startswith('Pss:'):\n"
       "        pss = int(line.split()[1])\n"
       "print(pss)")


def fork_ms(sb, n=20):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        br = sb.fork()
        ts.append((time.perf_counter() - t) * 1e3)
        br.close()
    return round(statistics.median(ts), 2)


def main():
    big = Limits(memory_mb=8192, tmpfs_mb=512, file_mb=256, processes=256)
    with Zygote(preload=["json", "math"]) as z:
        print(json.dumps({"bench": "fork", "cpus": os.cpu_count(), "features": z.features}), flush=True)

        rows = []
        for mb in (0, 64, 256, 1024):
            with z.spawn(big) as sb:
                sb.exec(f"state = bytearray({mb} << 20)\nfor i in range(0, len(state), 4096): state[i] = 1")
                rows.append({"python_state_mb": mb, "fork_ms": fork_ms(sb)})
        print(json.dumps({"fork_vs_state": rows}), flush=True)

        rows = []
        for files, kb in ((0, 0), (100, 10), (100, 160), (1000, 16)):
            with z.spawn(big) as sb:
                if files:
                    sb.exec(f"for i in range({files}): open(f'f{{i}}.py', 'wb').write(b'x' * {kb * 1024})")
                rows.append({"workspace_files": files, "workspace_mb": round(files * kb / 1024, 2), "fork_ms": fork_ms(sb, 10)})
        print(json.dumps({"fork_vs_workspace": rows}), flush=True)

        with z.spawn(big) as sb:
            sb.exec("state = bytearray(256 << 20)\nfor i in range(0, len(state), 4096): state[i] = 1")
            parent = int(sb.exec(PSS)["stdout"])
            branches = [sb.fork() for _ in range(8)]
            each = [int(b.exec(PSS)["stdout"]) for b in branches]
            touched = []
            for b in branches[:2]:  # a branch that writes its copy pays for it
                b.exec("for i in range(0, len(state), 4096): state[i] = 2")
            touched = [int(b.exec(PSS)["stdout"]) for b in branches[:2]]
            for b in branches:
                b.close()
        print(json.dumps({"memory_8_branches_of_256mb": {
            "parent_pss_mb_after_forking": round(parent / 1024, 1),
            "branch_pss_mb_median": round(statistics.median(each) / 1024, 1),
            "total_pss_mb_9_processes": round((parent + sum(each)) / 1024, 1),
            "if_copied_mb": 9 * 256,
            "branch_after_writing_all_pss_mb": round(statistics.median(touched) / 1024, 1)}}), flush=True)

        setup = ("import json, re, collections, statistics, decimal, fractions\n"
                 "table = {i: str(i) * 3 for i in range(400000)}\n"
                 "index = collections.Counter(len(v) for v in table.values())\n")
        with z.spawn(big) as sb:
            t = time.perf_counter()
            assert sb.exec(setup)["ok"]
            build = (time.perf_counter() - t) * 1e3
            replay = []
            for _ in range(5):
                t = time.perf_counter()
                with z.spawn(big) as fresh:
                    assert fresh.exec(setup)["ok"]
                    assert fresh.exec("print(len(table))")["stdout"] == "400000\n"
                replay.append((time.perf_counter() - t) * 1e3)
            branch = []
            for _ in range(5):
                t = time.perf_counter()
                with sb.fork() as br:
                    assert br.exec("print(len(table))")["stdout"] == "400000\n"
                branch.append((time.perf_counter() - t) * 1e3)
        print(json.dumps({"branch_vs_replay": {
            "build_state_ms": round(build, 1), "replay_ms_median": round(statistics.median(replay), 1),
            "fork_ms_median": round(statistics.median(branch), 2),
            "speedup": round(statistics.median(replay) / statistics.median(branch), 1)}}), flush=True)


if __name__ == "__main__":
    main()
