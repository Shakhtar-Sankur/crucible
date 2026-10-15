"""The zygote: a warm Python process (the modules a task needs already imported) that
forks a sandbox per request. A fork copies its memory copy-on-write, so a sandbox starts
with everything loaded in a fraction of the time a new interpreter takes.

Protocol (on the socket inherited from the client, fd given by --fd):
  -> {"op": "spawn", "slot": int, "limits": {...}}
  <- {"pid": int}   with the new sandbox's socket attached (SCM_RIGHTS)
The sandbox then speaks runner.py's protocol on that socket. Its first messages are
{"init_pid": outer pid or null} and the runner's {"ready"}, sent with its credentials so the
client learns the runner's pid as it sees it.

With a PID namespace the runner is not the namespace's init: init is a separate process that
only reaps, so the runner (process 2) and the branches forked from it each live in their own
process group and can be killed alone, a branch outliving the sandbox it was forked from.
Killing init ends every process in the namespace; the client does that when the last sandbox
of the tree is closed."""

import argparse
import importlib
import os
import signal
import socket

from . import isolate, probe, runner, wire
from .config import Limits


def _child(sock, client, limits, slot, features):
    client.close()
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    os.setsid()
    layers, pidns = isolate.enter(limits, slot, features)
    if pidns:
        pid = os.fork()  # the next child is the namespace's init
        if pid:
            wire.send(sock, {"init_pid": pid})
            sock.close()
            os.waitpid(pid, 0)
            os._exit(0)
        layers = isolate.finish(limits, layers, as_pid1=True)
        if os.fork():
            _init(sock)
        os.setpgid(0, 0)
    else:
        wire.send(sock, {"init_pid": None})
        layers = isolate.finish(limits, layers, as_pid1=False)
    runner.serve(sock, limits, layers)
    os._exit(0)


def _init(sock):
    """The PID namespace's init: reaps every process that ends (the runner, branches whose
    parent is gone) and leaves when none is left, which ends the namespace."""
    sock.close()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            os._exit(0)
        except InterruptedError:
            continue


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", type=int, required=True)
    ap.add_argument("--preload", default="", help="comma-separated modules to import before forking")
    args = ap.parse_args(argv)
    client = socket.socket(fileno=args.fd)
    for m in filter(None, args.preload.split(",")):
        importlib.import_module(m)
    features = probe.features()
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)  # sandboxes are reaped automatically
    wire.send(client, {"hello": True, "features": features, "pid": os.getpid()})
    while True:
        try:
            msg, _ = wire.recv(client)
        except (ConnectionError, OSError):
            return
        if msg.get("op") != "spawn":
            wire.send(client, {"error": f"unknown op {msg.get('op')}"})
            continue
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        wire.enable_creds(ours)  # the runner's first message carries its pid
        pid = os.fork()
        if pid == 0:
            ours.close()
            try:
                _child(theirs, client, Limits.from_dict(msg["limits"]), msg["slot"], features)
            except BaseException as e:
                try:
                    wire.send(theirs, {"error": f"{type(e).__name__}: {e}"})
                finally:
                    os._exit(1)
        theirs.close()
        wire.send(client, {"pid": pid}, fds=[ours.fileno()])
        ours.close()


if __name__ == "__main__":
    main()
