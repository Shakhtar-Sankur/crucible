"""The zygote: a warm Python process (the modules a task needs already imported) that
forks a sandbox per request. A fork copies its memory copy-on-write, so a sandbox starts
with everything loaded in a fraction of the time a new interpreter takes.

Protocol (on the socket inherited from the client, fd given by --fd):
  -> {"op": "spawn", "slot": int, "limits": {...}}
  <- {"pid": int}   with the new sandbox's socket attached (SCM_RIGHTS)
The sandbox then speaks runner.py's protocol on that socket; its first message is
{"runner_pid": outer pid}, the process to kill to end it (with a PID namespace, the
namespace's init: killing it ends every process in the sandbox)."""

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
            wire.send(sock, {"runner_pid": pid})
            sock.close()
            _, status = os.waitpid(pid, 0)
            os._exit(0)
    else:
        wire.send(sock, {"runner_pid": os.getpid()})
    layers = isolate.finish(limits, layers, as_pid1=pidns)
    runner.serve(sock, limits, layers)
    os._exit(0)


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
