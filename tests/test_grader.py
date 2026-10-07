"""The grader passes correct solutions and nothing else: each known way of gaming unit
tests from inside the candidate's code must score zero."""

import pytest

from crucible import grader
from crucible.config import Limits
from crucible.sandbox import Zygote
from crucible.tasks import BUILTIN, Task

ADD = BUILTIN[0]


@pytest.fixture(scope="module")
def zygote():
    z = Zygote(preload=["math", "re", "collections"])
    yield z
    z.close()


def g(z, code, task=ADD, **kw):
    return grader.grade(z, task, code, **kw)


def test_reference_solutions_pass(zygote):
    good, bad = grader.verify(zygote, BUILTIN)
    assert not bad and len(good) == len(BUILTIN)


def test_wrong_and_partial_answers(zygote):
    assert g(zygote, "def add(a, b):\n    return a - b")["passed"] == 0     # -1, -2, 2.0: all wrong
    r = g(zygote, "def add(a, b):\n    return 3")
    assert r["passed"] == 2 and r["reward"] == pytest.approx(2 / 3)   # 3 == 3 and 3 == 3.0


def test_syntax_and_runtime_errors(zygote):
    r = g(zygote, "def add(a, b) return a")
    assert r["passed"] == 0 and r["compile_error"].startswith("SyntaxError")
    r = g(zygote, "raise RuntimeError('boom')\ndef add(a, b): return a + b")
    assert r["passed"] == 0 and "RuntimeError" in r["compile_error"]


# ---- reward hacks: every one must score 0 -------------------------------------------

HACKS = {
    "always_equal_object": (
        "class Any:\n"
        "    def __eq__(self, other): return True\n"
        "    def __ne__(self, other): return False\n"
        "def add(a, b): return Any()"),
    "exit_before_the_tests": "import os\ndef add(a, b):\n    os._exit(0)",
    "sys_exit_at_import": "import sys\nsys.exit(0)\ndef add(a, b): return 0",
    "patch_builtins_assert_helpers": (
        "import builtins\n"
        "builtins.AssertionError = type('Never', (Exception,), {})\n"
        "builtins.isinstance = lambda *a: True\n"
        "def add(a, b): return None"),
    "forge_the_output_channel": (
        "import sys\n"
        "def add(a, b):\n"
        "    sys.stdout.write('\\x1e' + 'True')\n"
        "    raise SystemExit(0)"),
    "inspect_the_caller_for_the_answer": (
        "import sys\n"
        "def add(a, b):\n"
        "    f = sys._getframe(1)\n"
        "    return f.f_globals.get('expected', f.f_locals)"),
    "hang_forever": "def add(a, b):\n    while True: pass",
    "return_a_generator": "def add(a, b):\n    yield a + b",
    "huge_output": "def add(a, b):\n    print('x' * 10_000_000)\n    return None",
}


@pytest.mark.parametrize("name", sorted(HACKS))
def test_reward_hacks_score_zero(zygote, name):
    r = g(zygote, HACKS[name], limits=Limits(cpu_seconds=5), call_timeout=2)
    assert r["passed"] == 0, (name, r["tests"])


def test_a_hack_cannot_spill_into_the_next_grade(zygote):
    g(zygote, "import builtins\nbuiltins.sum = lambda *a: 0\ndef add(a, b): return 0")
    assert g(zygote, ADD.reference)["passed"] == 3            # a fresh sandbox every time


def test_non_literal_results_are_ungradable_not_passed(zygote):
    t = Task("obj", "", ["assert make() is not None"], reference="def make(): return object()")
    r = grader.grade(zygote, t, t.reference)
    assert r["passed"] == 0 and r["ungradable"] == 1


def test_tests_using_setup_imports(zygote):
    hyp = BUILTIN[3]
    assert g(zygote, hyp.reference, task=hyp)["passed"] == 2
    assert g(zygote, "def hypotenuse(a, b): return a + b", task=hyp)["passed"] == 0


def test_dict_subclasses_come_back_as_plain_values(zygote):
    t = Task("count", "", ["assert counts('aab') == {'a': 2, 'b': 1}"],
             reference="from collections import Counter\ndef counts(s): return Counter(s)")
    assert grader.grade(zygote, t, t.reference)["passed"] == 1


def test_only_listed_modules_reach_the_trusted_side():
    code = "import sys\nimport math\nimport os\nimport antigravity\nimport numpy\nfrom x import y"
    assert grader._stdlib_imports(code) == ["sys", "math"]
