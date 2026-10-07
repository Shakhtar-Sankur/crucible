"""Coding tasks: a description, the imports the tests use, and assert-style tests.

MBPP (Austin et al., 2021; ~1,000 short Python problems, three asserts each) is the main
source; a few built-in tasks keep the tests self-contained."""

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class Task:
    task_id: str
    prompt: str                 # what the model is asked to write
    tests: list                 # assert statements, each one line of Python
    setup: str = ""             # imports the tests (and usually the solution) need
    reference: str = ""         # a known-good solution, used to verify the task itself
    entry_point: str = ""       # the function the tests call, when known
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


BUILTIN = [
    Task("add", "Write a function add(a, b) that returns a + b.",
         ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(2.5, 0.5) == 3.0"],
         reference="def add(a, b):\n    return a + b\n", entry_point="add"),
    Task("reverse", "Write a function reverse_words(s) that reverses the order of the words in s.",
         ["assert reverse_words('a b c') == 'c b a'", "assert reverse_words('one') == 'one'",
          "assert reverse_words('') == ''"],
         reference="def reverse_words(s):\n    return ' '.join(s.split()[::-1])\n", entry_point="reverse_words"),
    Task("primes", "Write a function primes_below(n) that returns the sorted list of primes less than n.",
         ["assert primes_below(10) == [2, 3, 5, 7]", "assert primes_below(2) == []",
          "assert len(primes_below(1000)) == 168"],
         reference=("def primes_below(n):\n"
                    "    sieve = [True] * max(n, 0)\n"
                    "    out = []\n"
                    "    for i in range(2, n):\n"
                    "        if sieve[i]:\n"
                    "            out.append(i)\n"
                    "            for j in range(i * i, n, i):\n"
                    "                sieve[j] = False\n"
                    "    return out\n"), entry_point="primes_below"),
    Task("hypot", "Write a function hypotenuse(a, b) for a right triangle with legs a and b.",
         ["assert math.isclose(hypotenuse(3, 4), 5.0)", "assert math.isclose(hypotenuse(5, 12), 13.0)"],
         setup="import math", reference="import math\ndef hypotenuse(a, b):\n    return math.hypot(a, b)\n",
         entry_point="hypotenuse"),
]


def mbpp_prompt(text, tests):
    """The usual MBPP prompt: the description plus one test, which fixes the function name."""
    return f"{text.strip()}\nYour code should pass this test:\n{tests[0]}"


def load_mbpp(path):
    """Tasks from a JSONL export of MBPP (fields as in the Hugging Face dataset
    google-research-datasets/mbpp, configs "full" or "sanitized")."""
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            text = r.get("text") or r.get("prompt")
            tests = list(r["test_list"])
            setup = "\n".join(r.get("test_imports") or []) or (r.get("test_setup_code") or "")
            out.append(Task(f"mbpp/{r['task_id']}", mbpp_prompt(text, tests), tests, setup=setup,
                            reference=r.get("code", ""), meta={"source": os.path.basename(path)}))
    return out


def download_mbpp(out_dir, config="full"):
    """Writes {split}.jsonl for every split of MBPP (needs huggingface_hub and pyarrow)."""
    from huggingface_hub import list_repo_files, hf_hub_download
    import pyarrow.parquet as pq
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for f in list_repo_files("google-research-datasets/mbpp", repo_type="dataset"):
        if f.startswith(config + "/") and f.endswith(".parquet"):
            split = os.path.basename(f).split("-")[0]
            local = hf_hub_download("google-research-datasets/mbpp", f, repo_type="dataset")
            dst = os.path.join(out_dir, f"{split}.jsonl")
            with open(dst, "w") as g:
                for row in pq.read_table(local).to_pylist():
                    g.write(json.dumps(row) + "\n")
            paths[split] = dst
    return paths
