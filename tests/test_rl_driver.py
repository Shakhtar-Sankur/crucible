"""The MBPP GRPO driver end to end on the CPU with ratchet's tiny test model and stand-in
tokenizer (skipped without a ratchet checkout with relay built: CRUCIBLE_RATCHET, or
../ratchet). Plumbing, not accuracy; and the code extraction."""

import json
import os

import pytest

from crucible.rl.mbpp_grpo import extract_code

RATCHET = os.environ.get("CRUCIBLE_RATCHET",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ratchet"))
TINY = os.path.join(RATCHET, "relay", "tests", "fixtures", "chat-tiny")


def test_extract_code():
    assert extract_code("Here:\n```python\ndef f(x):\n    return x\n```\nDone.") == "def f(x):\n    return x\n"
    two = "```\npip install x\n```\nthen\n```py\nimport math\ndef g(): pass\n```"
    assert extract_code(two) == "import math\ndef g(): pass\n"
    assert extract_code("```python\ndef h(): return 1\n") == "def h(): return 1\n"   # cut off at max tokens
    assert extract_code("def k(): return 2") == "def k(): return 2"


@pytest.mark.skipif(not os.path.isdir(TINY), reason="no ratchet checkout with relay")
def test_train_and_eval_run_on_cpu(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from crucible.rl import mbpp_grpo
    monkeypatch.setenv("RATCHET_FAKE_TOKENIZER", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rows = {"train": 12, "test": 4, "validation": 2, "prompt": 2}
    for split, n in rows.items():
        with open(tmp_path / f"{split}.jsonl", "w") as f:
            for i in range(n):
                f.write(json.dumps({"task_id": f"{split}{i}", "text": f"Return {i}.",
                                    "code": f"def f():\n    return {i}\n",
                                    "test_list": [f"assert f() == {i}"], "test_setup_code": ""}) + "\n")
    out = tmp_path / "out.jsonl"
    mbpp_grpo.main(["train", "--model", TINY, "--mbpp", str(tmp_path), "--ratchet", RATCHET, "--out", str(out),
                    "--backend", "cpu", "--train-device", "cpu", "--relay-device", "0", "--steps", "3",
                    "--prompts", "2", "--group", "4", "--max-new", "8", "--eval-max-new", "8", "--kv-blocks", "256",
                    "--lr", "1e-3", "--grade-workers", "4"])
    recs = [json.loads(line) for line in open(out)]
    tasks = [r for r in recs if r.get("phase") == "tasks"]
    assert tasks[0]["test"] == 4 and tasks[1]["train"] == 16
    evals = [r for r in recs if r.get("phase") == "eval"]
    assert [e["when"] for e in evals] == ["before", "after"] and evals[0]["n"] == 4
    steps = [r for r in recs if "step" in r]
    assert len(steps) == 3
    for s in steps:   # the stand-in tokenizer writes no code: nothing passes either grader
        assert s["reward"] == 0.0 and s["usual_only"] == 0 and "reward_seconds" in s and s["answers"] == 8
