# crucible

Sandboxed environments for training coding agents with reinforcement learning: every
rollout runs its code in its own isolated, disposable sandbox, forked in milliseconds
from a warm template, and the reward comes from tests run inside it. Built from scratch on
Linux namespaces, resource limits and `fork()`, no virtual machines (it runs where KVM is
not available, such as Kaggle and most CI), and trained end to end with
[ratchet](https://github.com/Shakhtar-Sankur/ratchet), my GRPO stack.

```
            ┌────────────── zygote: Python with the task's modules already imported ──────────────┐
 request ──▶│ fork ─▶ own user · mount/net/pid/ipc/uts namespaces · private tmpfs · limits ·      │
            │         no capabilities · no_new_privs ─▶ sandbox (persistent Python state)          │──▶ socket to the client
            └──────────────────────────────────────────────────────────────────────────────────────┘
```

Agent RL runs the model's code many thousands of times per training step, so sandboxes
must be cheap, isolated and, for multi-step tasks, forkable: labs now snapshot a sandbox
mid-episode and branch several attempts from it (DeltaBox, Branching Policy Optimization,
2026). crucible is an open implementation of those ideas on commodity Linux, with the
environments speaking [OpenEnv](https://github.com/huggingface/OpenEnv)'s interface.

## Plan

| | Milestone | State |
|---|---|---|
| M0 | Sandbox runtime: per-sandbox user, namespaces, private tmpfs, limits, no capabilities, seccomp filter; a warm zygote that forks sandboxes; a probe of what the machine allows; tests that each limit holds | done |
| M1 | Fork a live sandbox (memory copy-on-write, its files copied), for branching | done |
| M2 | OpenEnv-compatible server and a coding environment (hidden unit tests give the reward), and a grader that reward hacks cannot pass | done |
| M3 | ratchet's GRPO on coding problems with sandboxed rewards, 2× T4: pass rate before/after, GPU idle time | done |
| M4 | Branching rollouts from mid-episode snapshots: cost against replaying from the start | |
| M5 | Write-up | |

## M0: the sandbox

`Zygote(preload=[...])` starts a Python process that imports the task's modules once;
`spawn(Limits(...))` forks it, and the child becomes a sandbox in this order:

1. its own Linux user (`200000 + slot`), so file permissions keep it out of everything
   owned by root and out of other sandboxes, and the process limit counts only its own;
2. mount, network, IPC, UTS and PID namespaces: no network interfaces up (no network at
   all), a private tmpfs over `/tmp`, `/var/tmp` and `/dev/shm` (its workspace, size-limited),
   its own `/proc` in which it is process 1 and sees nothing else, and killing it ends every
   process it started;
3. limits: address space, CPU time, processes, open files, file size, no core dumps;
4. all capabilities dropped and `no_new_privs` set, so no setuid program gives any back;
5. a seccomp-BPF filter (`crucible/seccomp.py`): the kernel refuses internet sockets (only
   local Unix sockets remain), acting on other processes (ptrace, process_vm_*), and
   changing the system or its own confinement (mount, unshare, setns, modules, bpf, keyctl,
   io_uring, ...). Behind the namespaces it is a second line of defence; where namespaces
   are not allowed, it is what keeps the sandbox off the network.

Each layer is used only if the machine allows it (`python -m crucible.probe`), and every
sandbox reports which layers it actually has. What two machines allow:

| Layer | this dev machine, GitHub Actions | Kaggle notebook |
|---|---|---|
| own user, limits, no capabilities, no_new_privs | yes | yes |
| seccomp filter (no network, no ptrace/mount/unshare/...) | yes | yes |
| mount, network, PID, IPC, UTS namespaces; private tmpfs; own /proc | yes | **no** |

On Kaggle a sandbox is therefore off the network and out of other users' files, but shares
the machine's `/tmp` (its workspace is a directory only its user can open) and can list the
machine's processes. The client enforces wall-clock timeouts by
killing the sandbox. State persists across calls in one sandbox (a Python namespace), which
is what M1 forks.

`tests/test_sandbox.py` runs code into each limit: no network, memory, CPU time, wall
clock, processes, file size, writes outside its workspace, sandboxes seeing each other's
files or processes, the seccomp filter, and 640 spawns from 16 threads.

Start a sandbox, run `print(sum(range(100)))` and close it (`bench/spawn.py`, 4 CPUs,
Python 3.11; the fresh interpreter has no isolation at all):

| | median | p95 | per second, 16 in flight |
|---|---|---|---|
| crucible (forked from the zygote, every layer on) | **5.3 ms** | 6.3 ms | **526** |
| a fresh `python -c` | 19.8 ms | 26.4 ms | 188 |

With numpy preloaded a sandbox starts in 5.2 ms against 108 ms for a fresh interpreter
importing it (21×), and costs 2.5 MB of proportional memory, its pages shared with the
zygote until written.

On a Kaggle notebook (4 CPUs, Python 3.13; no namespaces there, seccomp on): **5.0 ms**
against 87 ms for a fresh interpreter (17×), **498 per second** against 26 with 16 in flight,
and with numpy preloaded 5.4 ms against 204 ms (38×). All tests pass there except the one
that needs namespaces (sandboxes not seeing each other's files and processes).

Not yet: CPU and memory accounting by cgroup rather than per-process limits.

## M1: forking a live sandbox

`branch = sandbox.fork()` returns a copy of the sandbox as it is now: its Python state
(the process is forked, so memory is shared copy-on-write until one side writes it) and its
workspace (copied). The two then go on independently, and either can be forked again.
Each branch is its own process group, killed alone on a timeout without touching its
parent or siblings. Inside a PID namespace a branch knows only its inner pid; it sends its
first message with `SCM_CREDENTIALS`, and the kernel translates the pid into the client's
view, the one the client can signal. Branches keep every limit and the seccomp filter
(`tests/test_fork.py`: state and files carried over, branches diverging, branches of
branches, a branch killed on a timeout while its siblings go on, 16 concurrent forks).

Measured on 4 CPUs (`bench/fork.py`):

| | |
|---|---|
| fork a sandbox with 0 / 64 / 256 / 1024 MB of Python state | 2.3 / 3.7 / 4.8 / 12.6 ms |
| ... with 1 MB in 100 files / 16 MB in 100 files / 16 MB in 1,000 files | 5.8 / 14.1 / 44.3 ms |
| 8 branches of a sandbox holding 256 MB: memory of all 9 processes | **508 MB** (2,304 MB if copied); a branch costs 31 MB until it writes, 258 MB after rewriting everything |
| reach a state that takes 90 ms to build (imports + a 400,000-entry table): fork it vs start a new sandbox and build it again | **4.5 ms vs 138 ms (30×)** |

The workspace copy is the expensive part for large workspaces (about 40 µs per file); a
copy-on-write file layer would remove it, and is not built. Threads started by the
sandbox's code are not carried into a branch (`fork()` copies only the calling thread).

## M2: a coding environment, and grading that cannot be gamed

RL against unit tests invites reward hacking: when the tests run in the same process as the
model's code, exiting early, returning an object whose `__eq__` always says True, or
printing a fake result scores as well as a right answer. In crucible's grader
(`crucible/grader.py`) they never share a process:

- the candidate's code runs in a sandbox, and only there;
- each test's `assert` is evaluated in the trusted process, which never runs candidate
  code: a call to a function the candidate defined is sent to the sandbox, and only a
  *value* comes back, as a Python literal parsed with `ast.literal_eval` (dict, list, set
  and tuple subclasses such as `Counter` are sent as their plain equivalents);
- anything that is not a clean value (an exception, a timeout, a dead sandbox, output that
  is not a literal) fails that test. The reward is the fraction of tests passed.

The environment (`crucible/envs/coding.py`) is an [OpenEnv](https://github.com/huggingface/OpenEnv)
environment served by OpenEnv's own `create_app` (`python -m crucible.envs.server --tasks
mbpp.jsonl`): `reset()` gives a task's prompt; a `run` step executes code in the episode's
sandbox, whose state persists between steps; a `submit` step grades in a fresh sandbox and
ends the episode. Sessions over `/ws` keep the episode; a trainer can grade a completion in
one `POST /step` by naming the task (`tests/test_openenv.py`).

Measured on all 974 MBPP tasks, 4 CPUs (`bench/grading.py`; Kaggle: `results/kaggle-cpu-m2-2026-10-07.txt`), against the usual grader (the
solution and its asserts in one Python process, exit status 0 is a pass):

| | crucible | usual |
|---|---|---|
| reference solutions that pass all their tests | 970 / 974 | 972 / 974 |
| tasks graded per second, one at a time | 86.7 | 74.7 |
| tasks graded per second, 8 in flight | 212.2 | 202.4 |
| same on Kaggle's CPU notebook (fresh Python: 86 ms there) | 72.9 / 130.9 | 11.9 / 27.0 |
| reward hacks that pass (of 9, `tests/test_grader.py`) | **0** | **4**: an always-equal object, exiting before the tests, a forged output, `sys.exit` at import |

The four MBPP tasks crucible cannot grade need values that are not literals: a custom
class as the result (601) or as a test's input (927, 367: `Node`), and a type as an
argument (533). They fail closed, as failed tests, and `grader.verify()` (each task's
reference solution through the grader) leaves them out of the training set. The usual
grader's two misses are reference solutions that fail their own tests.

## M3: GRPO on coding problems, rewarded in the sandbox

ratchet's GRPO trains Qwen2.5-Coder-0.5B-Instruct on MBPP's 471 gradable training tasks,
with every reward from crucible's grader (`crucible/rl/mbpp_grpo.py`). relay generates on
one T4 while the trainer updates on the other, one step ahead; each step is 8 tasks × 8
answers, graded in parallel in sandboxes. Every answer is also graded, in a sandbox, by the
usual grader, to count the rewards it would have given that crucible did not. One run of
100 steps on Kaggle's 2× T4 (`results/kaggle-m3-full-2026-10-08.txt`):

| | before | after 100 steps |
|---|---|---|
| pass@1 on 499 MBPP test tasks (greedy) | 34.9% (174) | **38.9% (194)** |
| tests passed | 41.8% | 47.2% |
| answers that do not parse | 12.4% (62) | 0.2% (1) |
| mean answer length | 117 tokens | 54 tokens |

What the numbers do and do not show:

- **The gain is +4.0 points from one run and one seed.** Unpaired, that is 1.3 standard
  errors; the eval did not keep per-task outcomes, so a paired test is not possible from
  this log. Treat it as a signal, not a result, until more seeds agree.
- **Much of it is format.** Unparseable answers fell from 62 to 1 and answers halved in
  length: the policy learned to emit just the function. How much of the 20 extra passes is
  better code rather than cleaner output, this run cannot separate.
- **No reward hacking appeared to stop.** crucible and the usual grader agreed on all 6,400
  training answers and on both evaluations (0 usual-only passes). At 0.5B parameters and
  100 steps the policy never found a hack, so this run does not exercise the grader's
  defences; M2's adversarial tests do.
- **The trainer is the bottleneck.** On the logged steps the trainer's GPU is busy 95–99%
  of each step and the generator's a median 64% (40–86%); a step takes a median 22.3 s.
  Grading both ways takes a median 2.9 s per 64 answers and overlaps training.
- **Fewer answers carry signal as training goes on.** A group whose 8 answers all score
  the same has no advantage to learn from: trained samples per step fall from 40–56 early
  to 16–24 by steps 60–90.
- The trainer–engine log-probability gap was measured only on step 1 (max 0.011), the one
  step with fresh samples; one step ahead, every later sample is one update behind.

## Run it

```
sudo python -m pytest tests          # namespaces and per-sandbox users need root
sudo python bench/spawn.py
sudo python bench/fork.py
sudo python bench/grading.py <dir with MBPP's *.jsonl>   # python -c 'from crucible.tasks import download_mbpp; download_mbpp("mbpp")'
pip install openenv && python -m crucible.envs.server   # the coding environment over HTTP/WebSocket
```
Kaggle: `!cd /tmp && rm -rf c && git clone -q --depth 1 https://github.com/Shakhtar-Sankur/crucible c && bash c/scripts/kaggle.sh`

M3 (GRPO on MBPP with ratchet, Kaggle "GPU T4 x2"): `!cd /tmp && rm -rf c && git clone -q --depth 1 https://github.com/Shakhtar-Sankur/crucible c && RUN=smoke bash c/scripts/kaggle_m3.sh` (`RUN=full` for the measured run)
