"""The command loop inside a sandbox. One persistent Python namespace per sandbox, so
state carries across steps (and across forks, in M1). Code's output is captured at the
file-descriptor level, so output from C extensions and subprocesses is caught too."""

import os
import sys
import time
import traceback

from . import wire

_CAPTURE = ".crucible"


def _capture_start():
    saved = (os.dup(1), os.dup(2))
    # in the workspace (its own directory even without a private /tmp)
    out = os.open(_CAPTURE + ".out", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
    err = os.open(_CAPTURE + ".err", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
    sys.stdout.flush(), sys.stderr.flush()
    os.dup2(out, 1), os.dup2(err, 2)
    return saved, (out, err)


def _capture_end(saved, files, limit):
    sys.stdout.flush(), sys.stderr.flush()
    os.dup2(saved[0], 1), os.dup2(saved[1], 2)
    texts = []
    for fd in files:
        size = os.lseek(fd, 0, os.SEEK_END)
        os.lseek(fd, 0, os.SEEK_SET)
        data = os.read(fd, min(size, limit))
        os.close(fd)
        texts.append((data.decode("utf-8", "replace"), size > limit))
    os.close(saved[0]), os.close(saved[1])
    return texts


def _exec(scope, code, limit):
    t = time.perf_counter()
    saved, files = _capture_start()
    error = None
    try:
        exec(compile(code, "<sandbox>", "exec"), scope)
    except SystemExit as e:
        error = {"type": "SystemExit", "message": str(e.code)}
    except BaseException as e:  # the sandbox's own failure, reported, never fatal to the loop
        error = {"type": type(e).__name__, "message": str(e)[:2000],
                 "traceback": traceback.format_exc()[-4000:]}
    (out, out_cut), (err, err_cut) = _capture_end(saved, files, limit)
    return {"ok": error is None, "error": error, "stdout": out, "stderr": err,
            "truncated": out_cut or err_cut, "seconds": time.perf_counter() - t}


def serve(sock, limits, layers):
    global _CAPTURE
    _CAPTURE = os.path.join(os.getcwd(), ".crucible")
    scope = {"__name__": "__sandbox__"}
    wire.send(sock, {"ready": True, "layers": layers})
    while True:
        try:
            msg, _ = wire.recv(sock)
        except (ConnectionError, OSError):
            os._exit(0)
        op = msg.get("op")
        if op == "exec":
            reply = _exec(scope, msg["code"], limits.output_bytes)
        elif op == "write":
            for path, text in msg["files"].items():
                with open(path, "w") as f:
                    f.write(text)
            reply = {"ok": True}
        elif op == "read":
            try:
                with open(msg["path"]) as f:
                    reply = {"ok": True, "text": f.read(limits.output_bytes)}
            except OSError as e:
                reply = {"ok": False, "error": {"type": type(e).__name__, "message": str(e)}}
        elif op == "ping":
            reply = {"ok": True, "pid": os.getpid(), "cwd": os.getcwd()}
        elif op == "close":
            os._exit(0)
        else:
            reply = {"ok": False, "error": {"type": "UnknownOp", "message": str(op)}}
        try:
            wire.send(sock, reply)
        except (ConnectionError, OSError):
            os._exit(0)
