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
| M0 | Sandbox runtime: per-sandbox user, namespaces, private tmpfs, limits, no capabilities; a warm zygote that forks sandboxes; a probe of what the machine allows; tests that each limit holds | done |
| M1 | Snapshot and fork a live sandbox (memory copy-on-write, its files copied), for branching | |
| M2 | OpenEnv-compatible server and a coding environment (hidden unit tests give the reward) | |
| M3 | ratchet's GRPO on coding problems with sandboxed rewards, 2× T4: pass rate before/after, GPU idle time | |
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
4. all capabilities dropped and `no_new_privs` set, so no setuid program gives any back.

Each layer is used only if the machine allows it (`python -m crucible.probe`), and every
sandbox reports which layers it actually has. The client enforces wall-clock timeouts by
killing the sandbox. State persists across calls in one sandbox (a Python namespace), which
is what M1 forks.

`tests/test_sandbox.py` runs code into each limit: no network, memory, CPU time, wall
clock, processes, file size, writes outside its workspace, sandboxes seeing each other's
files or processes, and 640 spawns from 16 threads.

Start a sandbox, run `print(sum(range(100)))` and close it (`bench/spawn.py`, 4 CPUs,
Python 3.11; the fresh interpreter has no isolation at all):

| | median | p95 | per second, 16 in flight |
|---|---|---|---|
| crucible (forked from the zygote, every layer on) | **5.3 ms** | 6.3 ms | **526** |
| a fresh `python -c` | 19.8 ms | 26.4 ms | 188 |

With numpy preloaded a sandbox starts in 5.2 ms against 108 ms for a fresh interpreter
importing it (21×), and costs 2.5 MB of proportional memory, its pages shared with the
zygote until written.

Not yet: a seccomp system-call filter (a second line of defence behind the namespaces),
and CPU/memory accounting by cgroup rather than per-process limits.

## Run it

```
sudo python -m pytest tests          # namespaces and per-sandbox users need root
sudo python bench/spawn.py
```
Kaggle: `!cd /tmp && rm -rf c && git clone -q --depth 1 https://github.com/Shakhtar-Sankur/crucible c && bash c/scripts/kaggle.sh`
