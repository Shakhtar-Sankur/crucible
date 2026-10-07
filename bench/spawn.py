"""How fast sandboxes start: forked from a warm zygote vs a fresh Python process.

For each way, the time from request to the answer of a trivial piece of code (start + run
+ close), median over runs; then throughput with many requests in flight; then memory: the
proportional set size (PSS: shared pages divided among the processes sharing them) of a
sandbox forked from a zygote that has numpy loaded. Prints JSON lines."""

import json
import os
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from crucible.config import Limits  # noqa: E402
from crucible.sandbox import Zygote  # noqa: E402

CODE = "print(sum(range(100)))"


def median_ms(fn, n):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts), sorted(ts)[int(0.95 * (n - 1))]


def throughput(fn, total, workers):
    left = [total]
    lock = threading.Lock()

    def work():
        while True:
            with lock:
                if left[0] == 0:
                    return
                left[0] -= 1
            fn()

    t = time.perf_counter()
    ts = [threading.Thread(target=work) for _ in range(workers)]
    for x in ts:
        x.start()
    for x in ts:
        x.join()
    return total / (time.perf_counter() - t)


def pss_kb(pid):
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except OSError:
        return None


def main(n=200, total=500, workers=16):
    preload = ["json", "math", "re", "collections", "itertools", "functools"]
    have_numpy = subprocess.run([sys.executable, "-c", "import numpy"], capture_output=True).returncode == 0
    with Zygote(preload=preload) as z:
        print(json.dumps({"bench": "spawn", "cpus": os.cpu_count(), "python": sys.version.split()[0],
                          "features": z.features}), flush=True)

        def zygote_one():
            with z.spawn(Limits()) as sb:
                assert sb.exec(CODE)["stdout"] == "4950\n"

        def fresh_one():
            r = subprocess.run([sys.executable, "-c", "import " + ",".join(preload) + "; " + CODE],
                               capture_output=True, text=True)
            assert r.stdout == "4950\n"

        rows = {}
        for name, fn in (("zygote_fork", zygote_one), ("fresh_python", fresh_one)):
            med, p95 = median_ms(fn, n)
            rows[name] = {"median_ms": round(med, 2), "p95_ms": round(p95, 2),
                          "per_second_16_in_flight": round(throughput(fn, total, workers), 1)}
        rows["speedup_median"] = round(rows["fresh_python"]["median_ms"] / rows["zygote_fork"]["median_ms"], 1)
        print(json.dumps({"start_run_close": rows}), flush=True)

    if have_numpy:
        with Zygote(preload=preload + ["numpy"]) as z:
            sbs = [z.spawn(Limits(memory_mb=2048)) for _ in range(20)]
            for sb in sbs:
                assert sb.exec("import numpy as np; a = np.ones(10)")["ok"]
            pss = [pss_kb(sb.pid) for sb in sbs]
            zpss = pss_kb(z.proc.pid)
            med_np, _ = median_ms(lambda: z.spawn(Limits(memory_mb=2048)).close(), 50)
            for sb in sbs:
                sb.close()
        t = time.perf_counter()
        subprocess.run([sys.executable, "-c", "import numpy"], check=True)
        fresh_np = (time.perf_counter() - t) * 1e3
        print(json.dumps({"with_numpy_preloaded": {
            "zygote_spawn_median_ms": round(med_np, 2), "fresh_python_import_numpy_ms": round(fresh_np, 1),
            "sandbox_pss_kb_median": statistics.median([p for p in pss if p]) if any(pss) else None,
            "zygote_pss_kb": zpss}}), flush=True)


if __name__ == "__main__":
    main()
