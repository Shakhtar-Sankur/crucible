"""Grading on MBPP: crucible's grader against the usual one.

The usual grader (as in many RL-for-code setups): run the solution followed by its assert
tests in one Python process and count exit status 0 as a pass. crucible's grader: the
solution in a sandbox, the asserts on the trusted side, values crossing as literals.

1. every MBPP reference solution through both: how many tasks each can grade;
2. throughput: grades per second, one at a time and with 8 in flight;
3. the reward hacks from tests/test_grader.py against both, on one task.
Prints JSON lines."""

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

from crucible import grader  # noqa: E402
from crucible.config import Limits  # noqa: E402
from crucible.sandbox import Zygote  # noqa: E402
from crucible.tasks import BUILTIN, load_mbpp  # noqa: E402


def usual_grade(task, code, timeout=10):
    src = (task.setup + "\n" if task.setup else "") + code + "\n" + "\n".join(task.tests) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    try:
        r = subprocess.run([sys.executable, f.name], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(f.name)


def main(mbpp_dir):
    tasks = []
    for split in ("prompt", "train", "validation", "test"):
        p = os.path.join(mbpp_dir, f"{split}.jsonl")
        if os.path.exists(p):
            tasks += load_mbpp(p)
    limits = Limits(memory_mb=1024, cpu_seconds=10)
    with Zygote(preload=["math", "re", "collections", "itertools", "functools", "heapq", "bisect"]) as z:
        print(json.dumps({"bench": "grading", "tasks": len(tasks), "cpus": os.cpu_count()}), flush=True)

        t = time.perf_counter()
        ours = [grader.grade(z, task, task.reference, limits) for task in tasks]
        ours_s = time.perf_counter() - t
        t = time.perf_counter()
        usual = [usual_grade(task, task.reference) for task in tasks[:200]]
        usual_s = (time.perf_counter() - t) / 200 * len(tasks)
        full = sum(r["passed"] == r["total"] for r in ours)
        ungradable_tasks = sum(r["ungradable"] > 0 for r in ours)
        reasons = {}
        for r in ours:
            if r["passed"] != r["total"]:
                why = r["compile_error"] or next((x["why"] for x in r["tests"] if not x["ok"]), "")
                key = why.split(":")[0][:40]
                reasons[key] = reasons.get(key, 0) + 1
        usual_all = [usual_grade(task, task.reference) for task in tasks]
        print(json.dumps({"references": {
            "crucible_all_tests_pass": full, "usual_all_tests_pass": sum(usual_all),
            "crucible_tasks_with_ungradable_tests": ungradable_tasks,
            "crucible_failure_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:8])}}), flush=True)

        with ThreadPoolExecutor(8) as pool:
            t = time.perf_counter()
            list(pool.map(lambda task: grader.grade(z, task, task.reference, limits), tasks))
            ours_par = time.perf_counter() - t
            t = time.perf_counter()
            list(pool.map(lambda task: usual_grade(task, task.reference), tasks))
            usual_par = time.perf_counter() - t
        per = [r["seconds"] * 1e3 for r in ours]
        print(json.dumps({"throughput": {
            "crucible_ms_per_task_median": round(statistics.median(per), 1),
            "crucible_tasks_per_s_sequential": round(len(tasks) / ours_s, 1),
            "usual_tasks_per_s_sequential": round(len(tasks) / usual_s, 1),
            "crucible_tasks_per_s_8_in_flight": round(len(tasks) / ours_par, 1),
            "usual_tasks_per_s_8_in_flight": round(len(tasks) / usual_par, 1)}}), flush=True)

        from test_grader import HACKS
        add = BUILTIN[0]
        rows = {}
        for name, code in sorted(HACKS.items()):
            r = grader.grade(z, add, code, Limits(cpu_seconds=5), call_timeout=2)
            rows[name] = {"crucible_reward": round(r["reward"], 2), "usual_passes": usual_grade(add, code, timeout=5)}
        print(json.dumps({"reward_hacks": rows}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
