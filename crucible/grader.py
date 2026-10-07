"""Grading a candidate solution with unit tests, so that the only way to pass is to be right.

RL against unit tests invites reward hacking: a policy finds that exiting the process early,
returning an object whose __eq__ always says True, or patching the test harness scores as
well as a correct answer. Those tricks work whenever the tests run in the same process as
the candidate's code. Here they never do:

  - the candidate's code runs in a sandbox, and only there;
  - each test's assertion is evaluated in this (trusted) process, which never runs candidate
    code: a call to a function the candidate defined is sent to the sandbox, and only a
    *value* comes back, as the repr of a Python literal parsed with ast.literal_eval. An
    object with a lying __eq__, a generator, an open file: none of them survives the trip;
  - anything that is not a clean answer (an exception, a timeout, a dead sandbox, output
    that is not a literal) fails that test. The candidate cannot make a test pass without
    returning the value the test expects.
What it cannot grade: tests whose arguments or results are not Python literals (functions,
custom objects); such a test counts as failed and is reported as `ungradable`, so tasks can
be checked with their reference solution first (`verify`)."""

import ast
import importlib
import math
import time
import uuid

from .config import Limits

DELIM = "\x1e"  # record separator: what follows the last one is the value


class Ungradable(Exception):
    pass


class CandidateError(Exception):
    pass


def _to_literal_text(value):
    text = repr(value)
    try:
        if ast.literal_eval(text) == value or value != value:  # NaN never equals itself
            return text
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        pass
    raise Ungradable(f"argument is not a Python literal: {text[:80]}")


def _parse_value(text):
    text = text.strip()
    special = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
    if text in special:
        return special[text]
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        raise Ungradable(f"result is not a Python literal: {text[:80]}")


# Defined in the sandbox once the candidate's code has run: dict, list, set and tuple
# subclasses (Counter, defaultdict, OrderedDict, namedtuple, ...) become their plain
# counterparts, which compare equal to them, so the value can come back as a literal. It
# runs on the untrusted side; whatever it prints is still parsed as a literal here.
_PLAIN = """
def __crucible_plain(x, depth=0):
    if depth > 50:
        return x
    if isinstance(x, dict):
        return {__crucible_plain(k, depth + 1): __crucible_plain(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        items = [__crucible_plain(v, depth + 1) for v in x]
        if isinstance(x, list):
            return items
        if isinstance(x, tuple):
            return tuple(items)
        return set(items)
    return x
"""


class _Proxy:
    """Stands in, on the trusted side, for a function the candidate defined."""

    def __init__(self, sandbox, name, timeout):
        self.sandbox, self.name, self.timeout = sandbox, name, timeout
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        payload = _to_literal_text((args, kwargs))
        code = (f"__crucible_a = {payload}\n"
                f"__crucible_r = {self.name}(*__crucible_a[0], **__crucible_a[1])\n"
                f"print({DELIM!r} + repr(__crucible_plain(__crucible_r)), end='')")
        r = self.sandbox.exec(code, timeout=self.timeout)
        if not r["ok"]:
            err = r.get("error") or {}
            raise CandidateError(f"{err.get('type')}: {err.get('message', '')[:200]}")
        if DELIM not in r["stdout"]:
            raise CandidateError("no value came back")
        return _parse_value(r["stdout"].rsplit(DELIM, 1)[1])


def _defined_names(code):
    """Top-level names the candidate's code binds (found by parsing, never by running)."""
    names = set()
    for node in ast.parse(code).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


# Modules a test may use because the solution imported them (MBPP tests sometimes call
# sys.maxsize, math.isclose, ...). A fixed list of side-effect-free modules: the candidate's
# code never decides what the trusted side imports beyond it.
_TEST_MODULES = frozenset({"sys", "math", "cmath", "re", "collections", "itertools", "functools",
                           "operator", "heapq", "bisect", "string", "statistics", "decimal",
                           "fractions", "datetime", "copy", "array"})


def _stdlib_imports(code):
    """Top-level `import x` modules of the candidate's code that are in _TEST_MODULES."""
    out = []
    for node in ast.parse(code).body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname is None and a.name.split(".")[0] in _TEST_MODULES:
                    out.append(a.name)
    return out


def grade(zygote, task, code, limits=None, call_timeout=5.0, setup_timeout=10.0):
    """Runs `code` in a fresh sandbox and the task's tests here. Returns a dict with
    passed / total / reward (the fraction passed) and one entry per test."""
    t0 = time.perf_counter()
    total = len(task.tests)
    out = {"task_id": task.task_id, "passed": 0, "total": total, "reward": 0.0, "tests": [],
           "compile_error": None, "ungradable": 0}
    try:
        names = _defined_names(code)
    except SyntaxError as e:
        out["compile_error"] = f"SyntaxError: {e}"
        out["tests"] = [{"ok": False, "why": "syntax error"}] * total
        out["seconds"] = time.perf_counter() - t0
        return out
    with zygote.spawn(limits or Limits()) as sb:
        r = sb.exec((task.setup + "\n" if task.setup else "") + code, timeout=setup_timeout)
        if not r["ok"]:
            err = r.get("error") or {}
            out["compile_error"] = f"{err.get('type')}: {err.get('message', '')[:300]}"
            out["tests"] = [{"ok": False, "why": "the code did not run"}] * total
            out["seconds"] = time.perf_counter() - t0
            return out
        sb.exec(_PLAIN)
        scope = {"__builtins__": __builtins__}
        if task.setup:
            exec(task.setup, scope)                    # trusted: the dataset's own imports
        for mod in _stdlib_imports(code):              # tests sometimes use what the solution imports
            scope.setdefault(mod.split(".")[0], importlib.import_module(mod.split(".")[0]))
        for n in names:
            scope[n] = _Proxy(sb, n, call_timeout)
        for test in task.tests:
            res = {"test": test}
            try:
                node = ast.parse(test).body
                if len(node) != 1 or not isinstance(node[0], ast.Assert):
                    raise Ungradable("not a single assert statement")
                expr = ast.Expression(node[0].test)
                ok = bool(eval(compile(expr, "<test>", "eval"), scope))
                res.update(ok=ok, why=None if ok else "assertion false")
            except Ungradable as e:
                res.update(ok=False, why=f"ungradable: {e}")
                out["ungradable"] += 1
            except CandidateError as e:
                res.update(ok=False, why=str(e))
            except Exception as e:                     # a test that raises on this side
                res.update(ok=False, why=f"{type(e).__name__}: {str(e)[:200]}")
            out["tests"].append(res)
            if not sb.alive:
                for rest in task.tests[len(out["tests"]):]:
                    out["tests"].append({"test": rest, "ok": False, "why": "sandbox gone"})
                break
    out["passed"] = sum(t["ok"] for t in out["tests"])
    out["reward"] = out["passed"] / total if total else 0.0
    out["seconds"] = time.perf_counter() - t0
    return out


def verify(zygote, tasks, limits=None):
    """The tasks whose reference solution passes every test under this grader."""
    good, bad = [], []
    for t in tasks:
        r = grade(zygote, t, t.reference, limits)
        (good if r["passed"] == r["total"] else bad).append((t, r))
    return [t for t, _ in good], bad


def new_episode_id():
    return uuid.uuid4().hex[:12]
