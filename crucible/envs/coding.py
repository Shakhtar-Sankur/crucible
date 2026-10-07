"""crucible's coding environment, as an OpenEnv environment (pip install openenv).

An episode is one task. reset() returns its prompt; step() takes a CodeAction:
  kind="run"     executes code in the episode's sandbox (state persists between runs) and
                 returns stdout / stderr / error; no reward, the episode goes on;
  kind="submit"  grades the code with the task's hidden tests in a fresh sandbox
                 (crucible.grader: the tests never run alongside the candidate's code);
                 reward = fraction of tests passed; the episode ends.
Over OpenEnv's stateless HTTP endpoints, where every request gets a new environment, a
submit can name its task (CodeAction.task_id), which is how a trainer grades a completion
in one call. Sessions over /ws keep the episode (and its sandbox) between steps."""

import random
import threading
from typing import Any, List, Literal, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

from .. import grader
from ..config import Limits
from ..sandbox import Zygote
from ..tasks import BUILTIN, Task


class CodeAction(Action):
    code: str = Field(description="Python source")
    kind: Literal["run", "submit"] = Field(default="submit", description="run: execute; submit: grade")
    task_id: Optional[str] = Field(default=None, description="for one-shot grading without reset()")


class CodeObservation(Observation):
    task_id: str = ""
    prompt: str = ""
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    tests_passed: int = 0
    tests_total: int = 0
    test_results: List[Any] = Field(default_factory=list)


class CodeState(State):
    task_id: str = ""
    runs: int = 0
    submitted: bool = False


_zygote = None
_zygote_lock = threading.Lock()


def shared_zygote():
    """One warm zygote per server process, shared by every session."""
    global _zygote
    with _zygote_lock:
        if _zygote is None:
            _zygote = Zygote(preload=["math", "re", "collections", "itertools", "functools", "heapq", "bisect"])
        return _zygote


class CodingEnvironment(Environment[CodeAction, CodeObservation, CodeState]):
    SUPPORTS_CONCURRENT_SESSIONS = True     # each session has its own sandboxes
    TASKS: List[Task] = BUILTIN
    LIMITS = Limits(memory_mb=512, cpu_seconds=10)
    MAX_RUNS = 8

    def __init__(self):
        super().__init__()
        self._by_id = {t.task_id: t for t in self.TASKS}
        self._task = None
        self._sandbox = None
        self._state = CodeState()

    def _close_sandbox(self):
        if self._sandbox is not None:
            self._sandbox.close()
            self._sandbox = None

    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs) -> CodeObservation:
        self._close_sandbox()
        if task_id is not None:
            task = self._by_id[task_id]
        else:
            task = random.Random(seed).choice(self.TASKS)
        self._task = task
        self._state = CodeState(episode_id=episode_id or grader.new_episode_id(), task_id=task.task_id)
        return CodeObservation(task_id=task.task_id, prompt=task.prompt, tests_total=len(task.tests))

    def _episode_sandbox(self):
        if self._sandbox is None or not self._sandbox.alive:
            self._sandbox = shared_zygote().spawn(self.LIMITS)
            if self._task.setup:
                self._sandbox.exec(self._task.setup)
        return self._sandbox

    def step(self, action: CodeAction, timeout_s=None, **kwargs) -> CodeObservation:
        if self._task is None:
            if action.task_id is None or action.task_id not in self._by_id:
                return CodeObservation(done=True, reward=0.0, error="no task: call reset() or set task_id")
            self.reset(task_id=action.task_id)
        task, st = self._task, self._state
        st.step_count += 1
        if action.kind == "run":
            st.runs += 1
            r = self._episode_sandbox().exec(action.code, timeout=timeout_s or 10.0)
            err = r.get("error")
            done = st.runs >= self.MAX_RUNS
            return CodeObservation(task_id=task.task_id, stdout=r.get("stdout", ""), stderr=r.get("stderr", ""),
                                   error=f"{err['type']}: {err.get('message', '')}" if err else None,
                                   done=done, reward=0.0 if done else None, tests_total=len(task.tests))
        g = grader.grade(shared_zygote(), task, action.code, limits=self.LIMITS)
        st.submitted = True
        self._close_sandbox()
        return CodeObservation(task_id=task.task_id, done=True, reward=g["reward"], error=g["compile_error"],
                               tests_passed=g["passed"], tests_total=g["total"],
                               test_results=[{"ok": t["ok"], "why": t.get("why")} for t in g["tests"]],
                               metadata={"grade_seconds": g["seconds"], "ungradable": g["ungradable"]})

    @property
    def state(self) -> CodeState:
        return self._state

    def close(self):
        self._close_sandbox()


def environment_class(tasks, limits=None, max_runs=8):
    """A CodingEnvironment class bound to these tasks (OpenEnv's create_app takes a class
    and reads SUPPORTS_CONCURRENT_SESSIONS from it)."""
    return type("BoundCodingEnvironment", (CodingEnvironment,),
                {"TASKS": list(tasks), "LIMITS": limits or CodingEnvironment.LIMITS, "MAX_RUNS": max_runs})
