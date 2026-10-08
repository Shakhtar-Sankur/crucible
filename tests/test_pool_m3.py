"""scripts/pool_m3.py: pooling seeds from the per-task bit strings."""

import importlib.util
import os

import pytest

from crucible.rl.mbpp_grpo import mcnemar_p

_spec = importlib.util.spec_from_file_location(
    "pool_m3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "pool_m3.py"))
pool_m3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pool_m3)


def _run(before, after):
    return {"ids": "abc", "before": [int(c) for c in before], "after": [int(c) for c in after]}


def test_one_seed_matches_mcnemar():
    # With one seed the sign-flip test is McNemar's test on the discordant tasks.
    before = "0" * 20 + "1" * 6 + "0" * 10
    after = "1" * 20 + "0" * 6 + "0" * 10
    s = pool_m3.pool([_run(before, after)])
    assert (s["fixed_each"], s["broke_each"]) == ([20], [6])
    assert abs(s["p"] - mcnemar_p(6, 20)) < 0.01


def test_counts_across_seeds():
    s = pool_m3.pool([_run("0011", "1010"), _run("0011", "1110")])
    assert s["after_each"] == [0.5, 0.75]
    assert s["after_mean"] == 0.625
    assert (s["fixed_in_all"], s["broke_in_all"]) == (1, 1)


def test_rejects_different_starting_results():
    with pytest.raises(SystemExit):
        pool_m3.pool([_run("0011", "1010"), _run("0111", "1110")])
