"""Client side: start a zygote, spawn sandboxes from it, run code in them.

    with Zygote(preload=["json", "math"]) as z:
        sb = z.spawn(Limits(memory_mb=256))
        r = sb.exec("print(sum(range(10)))", timeout=5)   # r["stdout"] == "45\\n"
        sb.close()

Wall-clock timeouts are enforced here: a call that does not answer in time gets its
sandbox killed (the whole PID namespace when there is one) and reports timeout=True."""

import itertools
import os
import signal
import socket
import subprocess
import sys
import threading
import time

from . import wire
from .config import Limits


class SandboxError(RuntimeError):
    pass


class Sandbox:
    def __init__(self, sock, runner_pid, layers, spawn_seconds):
        self.sock, self.pid, self.layers = sock, runner_pid, layers
        self.spawn_seconds = spawn_seconds
        self.alive = True

    def _call(self, msg, timeout):
        if not self.alive:
            raise SandboxError("sandbox is closed")
        self.sock.settimeout(timeout)
        try:
            wire.send(self.sock, msg)
            reply, _ = wire.recv(self.sock)
            return reply
        except (socket.timeout, TimeoutError):
            self.kill()
            return {"ok": False, "timeout": True, "error": {"type": "Timeout", "message": f"no answer in {timeout}s"}}
        except (ConnectionError, OSError) as e:
            # the sandbox died (memory limit, CPU limit, or its own doing)
            self.kill()
            return {"ok": False, "died": True, "error": {"type": "SandboxDied", "message": str(e)}}

    def exec(self, code, timeout=10.0):
        return self._call({"op": "exec", "code": code}, timeout)

    def write(self, files, timeout=10.0):
        return self._call({"op": "write", "files": files}, timeout)

    def read(self, path, timeout=10.0):
        return self._call({"op": "read", "path": path}, timeout)

    def ping(self, timeout=5.0):
        return self._call({"op": "ping"}, timeout)

    def kill(self):
        if self.alive:
            self.alive = False
            try:
                os.kill(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.sock.close()

    def close(self):
        if self.alive:
            try:
                wire.send(self.sock, {"op": "close"})
            except OSError:
                pass
            self.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Zygote:
    def __init__(self, preload=(), python=sys.executable):
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.pathsep + env.get("PYTHONPATH", "")
        self.proc = subprocess.Popen([python, "-m", "crucible.zygote", "--fd", str(theirs.fileno()),
                                      "--preload", ",".join(preload)], pass_fds=[theirs.fileno()], env=env)
        theirs.close()
        self.sock = ours
        hello, _ = wire.recv(self.sock)
        self.features = hello["features"]
        self._lock = threading.Lock()
        self._slots = itertools.count()

    def spawn(self, limits=None, timeout=30.0):
        limits = limits or Limits()
        t = time.perf_counter()
        with self._lock:
            wire.send(self.sock, {"op": "spawn", "slot": next(self._slots) % 50000, "limits": limits.to_dict()})
            reply, fds = wire.recv(self.sock, want_fds=1)
        if not fds:
            raise SandboxError(f"zygote: {reply}")
        sock = socket.socket(fileno=fds[0])
        sock.settimeout(timeout)
        # Two messages: the process to kill ({"runner_pid"}) and {"ready"}. With a PID
        # namespace they come from two processes, so in either order.
        got = {}
        while not ("runner_pid" in got and "ready" in got):
            msg, _ = wire.recv(sock)
            if "error" in msg:
                sock.close()
                raise SandboxError(msg["error"])
            got.update(msg)
        return Sandbox(sock, got["runner_pid"], got["layers"], time.perf_counter() - t)

    def close(self):
        try:
            self.sock.close()
        finally:
            self.proc.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
