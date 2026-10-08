"""GRPO on MBPP with rewards from crucible's sandboxed grader (ratchet's trainer, relay's
rollouts).

  python -m crucible.rl.mbpp_grpo eval  --model M --mbpp D --ratchet R
  python -m crucible.rl.mbpp_grpo train --model M --mbpp D --ratchet R   # GPU 1 generates, GPU 0 trains

D holds MBPP's {train,validation,test,prompt}.jsonl (crucible.tasks.download_mbpp); R a
ratchet checkout with relay built. Training uses train + validation + prompt (474 tasks),
evaluation the 500 test tasks; tasks whose reference solution the grader cannot pass are
left out of both (grader.verify).

The reward of an answer is the fraction of its task's tests it passes, graded by
crucible.grader (tests on the trusted side, the code in a sandbox), all of a step's answers
in parallel. Every answer is also judged the usual way, in a sandbox: its code and the
asserts in one process, a pass on exit status 0. Answers the usual grader passes and
crucible does not are counted every step ("usual_only"): the reward the usual grader would
have given wrongly. Evaluation is greedy pass@1 (all tests pass) before and after."""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from .. import grader
from ..config import Limits
from ..sandbox import Zygote
from ..tasks import load_mbpp

_FENCE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)

INSTRUCTION = "Write a Python function for this task, in one ```python code block.\n\n{task}"


def extract_code(text):
    """The answer's code: the first fenced block that defines something, else the first
    fenced block, else an unterminated block's rest, else the whole text."""
    blocks = _FENCE.findall(text)
    for b in blocks:
        if re.search(r"^\s*(def|class) ", b, re.MULTILINE):
            return b
    if blocks:
        return blocks[0]
    m = re.search(r"```[ \t]*(?:python3?|py)?[ \t]*\n", text, re.IGNORECASE)
    return text[m.end():] if m else text


def load_tasks(mbpp_dir, splits):
    out = []
    for s in splits:
        p = os.path.join(mbpp_dir, f"{s}.jsonl")
        if os.path.exists(p):
            out += load_mbpp(p)
    return out


class Rewarder:
    """reward_fn for ratchet's GRPO: fraction of tests passed, crucible-graded; .batch grades
    a whole step in parallel and records the usual grader's verdict beside it."""

    def __init__(self, tok, zygote, limits, workers=8, compare=True, call_timeout=3.0):
        self.tok, self.z, self.limits = tok, zygote, limits
        self.pool = ThreadPoolExecutor(workers)
        self.compare, self.call_timeout = compare, call_timeout
        self.last = {}

    def _one(self, text, task):
        code = extract_code(text)
        g = grader.grade(self.z, task, code, self.limits, call_timeout=self.call_timeout, setup_timeout=5.0)
        usual = grader.grade_in_one_process(self.z, task, code, self.limits, timeout=5.0) if self.compare else None
        return g, usual

    def grade_texts(self, texts, tasks):
        t0 = time.perf_counter()
        res = list(self.pool.map(lambda a: self._one(*a), zip(texts, tasks)))
        n = max(len(res), 1)
        full = [g["passed"] == g["total"] for g, _ in res]
        self.last = {"reward_seconds": time.perf_counter() - t0, "pass_all": sum(full) / n,
                     "compile_errors": sum(g["compile_error"] is not None for g, _ in res) / n}
        if self.compare:
            usual = [u for _, u in res]
            self.last.update(usual_pass=sum(usual) / n,
                             usual_only=sum(u and not f for u, f in zip(usual, full)),
                             crucible_only=sum(f and not u for u, f in zip(usual, full)))
        return res

    def batch(self, samples):
        res = self.grade_texts([self.tok.decode(s.tokens) for s in samples], [s.answer for s in samples])
        return [g["reward"] for g, _ in res]

    def __call__(self, sample, task):
        return self.batch([sample])[0]


def evaluate(engine, tok, tasks, rewarder, max_new, relay):
    ids = [tok.prompt_ids(t.prompt) for t in tasks]
    for i, p in enumerate(ids):
        engine.add(10_000_000 + i, p, max_new_tokens=max_new, temperature=0.0)
    t0 = time.perf_counter()
    out = engine.run()
    gen_s = time.perf_counter() - t0
    comps = [out[10_000_000 + i] for i in range(len(tasks))]
    res = rewarder.grade_texts([tok.decode(c.tokens) for c in comps], tasks)
    n = max(len(tasks), 1)
    rec = {"n": len(tasks), "pass_at_1": sum(g["passed"] == g["total"] for g, _ in res) / n,
           "mean_test_fraction": sum(g["reward"] for g, _ in res) / n,
           "truncated": sum(c.finish == relay.FINISH_LENGTH for c in comps) / n,
           "mean_tokens": sum(len(c.tokens) for c in comps) / n, "generate_seconds": gen_s,
           **{k: v for k, v in rewarder.last.items() if k != "pass_all"}}
    # Each task's outcome, in the order of `tasks`, so two evaluations can be paired.
    rec["per_task"] = [{"id": t.task_id, "pass": g["passed"] == g["total"], "fraction": round(g["reward"], 4),
                        "parsed": g["compile_error"] is None} for t, (g, _) in zip(tasks, res)]
    return rec


def mcnemar_p(b, c):
    """Exact two-sided McNemar test: b tasks only one evaluation passes, c only the other."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(before, after):
    """Compares two evaluations of the same tasks, task by task. A task 'fixed' passes only
    after; it is split by why it failed before: its answer did not parse, or it parsed and
    failed tests. The bit strings (1 = passed, in task order) let runs be pooled later."""
    assert [x["id"] for x in before] == [x["id"] for x in after], "evaluations are of different tasks"
    both = sum(x["pass"] and y["pass"] for x, y in zip(before, after))
    fixed = [(x, y) for x, y in zip(before, after) if y["pass"] and not x["pass"]]
    broke = [(x, y) for x, y in zip(before, after) if x["pass"] and not y["pass"]]
    ids = ",".join(x["id"] for x in before)
    return {"n": len(before), "both_pass": both, "fixed": len(fixed),
            "fixed_had_not_parsed": sum(not x["parsed"] for x, _ in fixed),
            "fixed_had_failed_tests": sum(x["parsed"] for x, _ in fixed),
            "broke": len(broke), "broke_now_unparsed": sum(not y["parsed"] for _, y in broke),
            "neither": len(before) - both - len(fixed) - len(broke),
            "mcnemar_p": mcnemar_p(len(fixed), len(broke)),
            "ids_sha1": hashlib.sha1(ids.encode()).hexdigest()[:12],
            "before_bits": "".join("1" if x["pass"] else "0" for x in before),
            "after_bits": "".join("1" if y["pass"] else "0" for y in after)}


def _pieces(args):
    sys.path.insert(0, os.path.abspath(args.ratchet))
    from ratchet import gsm8k, relay

    class CodeTokenizer(gsm8k.ChatTokenizer):
        def prompt_ids(self, task_prompt):
            msgs = [{"role": "user", "content": INSTRUCTION.format(task=task_prompt)}]
            text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            return [int(t) for t in self.tok(text, add_special_tokens=False)["input_ids"]]

    return gsm8k, relay, CodeTokenizer(args.model)


def _verified(z, tasks, limits):
    good, bad = grader.verify(z, tasks, limits)
    return good, [t.task_id for t, _ in bad]


def run(args):
    gsm8k, relay, tok = _pieces(args)
    limits = Limits(memory_mb=1024, cpu_seconds=10)
    z = Zygote(preload=["math", "re", "collections", "itertools", "functools", "heapq", "bisect"])
    try:
        rewarder = Rewarder(tok, z, limits, workers=args.grade_workers, compare=not args.no_compare)
        test, dropped = _verified(z, load_tasks(args.mbpp, ["test"]), limits)
        if args.eval_limit:
            test = test[:args.eval_limit]
        gsm8k.log({"phase": "tasks", "test": len(test), "test_dropped": dropped}, args.out)
        model = relay.Model(args.model)
        engine = gsm8k.engine_for(model, args, args.relay_device)

        per_task = {}

        def ev(when):
            rec = evaluate(engine, tok, test, rewarder, args.eval_max_new, relay)
            per_task[when] = rec.pop("per_task")
            gsm8k.log({"phase": "eval", "when": when, **rec}, args.out)
            if args.out:
                with open(f"{args.out}.eval-{when}.json", "w") as f:
                    json.dump(per_task[when], f)

        if args.phase == "eval":
            ev("now")
            return
        train, tdropped = _verified(z, load_tasks(args.mbpp, ["train", "validation", "prompt"]), limits)
        gsm8k.log({"phase": "tasks", "train": len(train), "train_dropped": tdropped}, args.out)
        from ratchet.grpo import GRPO
        policy = gsm8k.policy_for(model, args, args.train_device)
        policy.checkpoint = True
        g = GRPO(model, engine, policy, rewarder, gsm8k.grpo_config(args, seed=args.seed))
        if not args.skip_eval:
            ev("before")
        problems = [(t.prompt, t) for t in train]
        gen = gsm8k.batches(problems, tok, args.prompts, args.steps, args.seed)
        t0 = time.perf_counter()
        mode = "split" + ("+async" if args.ahead else "")
        steps = g.run_async(gen) if args.ahead else (g.step(p, a) for p, a in gen)
        for m in steps:
            r = gsm8k.step_record(mode, m, {"answers": len(m["samples"]), **rewarder.last})
            r["elapsed"] = time.perf_counter() - t0
            gsm8k.log(r, args.out)
        if not args.skip_eval:
            ev("after")
            gsm8k.log({"phase": "paired", "seed": args.seed, **paired(per_task["before"], per_task["after"])}, args.out)
    finally:
        z.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["eval", "train"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--mbpp", required=True)
    ap.add_argument("--ratchet", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--train-device", default="cuda:0")
    ap.add_argument("--relay-device", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--old-logprobs", default="trainer", choices=["trainer", "rollout"])
    ap.add_argument("--sync", default="push", choices=["push", "reload"])
    ap.add_argument("--ahead", action="store_true", help="generate the next batch while training")
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--eval-max-new", type=int, default=512)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--kv-blocks", type=int, default=3000)
    ap.add_argument("--max-batch-tokens", type=int, default=1024)
    ap.add_argument("--max-seqs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-dtype", default="auto", choices=["auto", "fp16", "fp32"])
    ap.add_argument("--grade-workers", type=int, default=8)
    ap.add_argument("--no-compare", action="store_true", help="skip the usual grader's verdicts")
    args = ap.parse_args(argv)
    run(args)


if __name__ == "__main__":
    sys.exit(main())
