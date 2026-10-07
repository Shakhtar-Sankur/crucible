"""Which isolation layers this machine allows: each is tried in a throwaway child process.
Run `python -m crucible.probe` to print them."""

import json
import os
import sys

from . import _sys, seccomp

UID_BASE = 200000
_NS = {"mount": _sys.CLONE_NEWNS, "net": _sys.CLONE_NEWNET, "ipc": _sys.CLONE_NEWIPC,
       "uts": _sys.CLONE_NEWUTS, "pid": _sys.CLONE_NEWPID}


def _try(fn):
    pid = os.fork()
    if pid == 0:
        try:
            fn()
            os._exit(0)
        except BaseException:
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status) == 0


def _drop():
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(UID_BASE + 999)
        os.setuid(UID_BASE + 999)


def _userns(extra=0):
    _drop()
    uid = os.geteuid()
    _sys.unshare(_sys.CLONE_NEWUSER | extra)
    _sys.write_file("/proc/self/setgroups", "deny")
    _sys.write_file("/proc/self/uid_map", f"0 {uid} 1")
    _sys.write_file("/proc/self/gid_map", f"0 {uid} 1")


def features():
    root = os.geteuid() == 0
    f = {"root": root, "uid_base": UID_BASE, "kernel": os.uname().release}
    f["setuid"] = root and _try(_drop)
    f["ns_user"] = _try(_userns)
    for name, flag in _NS.items():
        def attempt(flag=flag, name=name):
            if f["ns_user"]:
                _userns(flag)
            elif root:
                _sys.unshare(flag)
            else:
                raise OSError("no way to create a namespace")
            if name == "mount":
                _sys.mount(None, "/", None, _sys.MS_REC | _sys.MS_PRIVATE)
                _sys.mount("tmpfs", "/tmp", "tmpfs", _sys.MS_NOSUID | _sys.MS_NODEV, "size=1m")
            if name == "pid":  # the next child is the namespace's init
                pid = os.fork()
                if pid == 0:
                    os._exit(0 if os.getpid() == 1 else 1)
                _, st = os.waitpid(pid, 0)
                if os.waitstatus_to_exitcode(st) != 0:
                    raise OSError("not pid 1")
        f[f"ns_{name}"] = _try(attempt)
    f["no_new_privs"] = _try(lambda: _sys.prctl(_sys.PR_SET_NO_NEW_PRIVS, 1))

    def filtered():
        _sys.prctl(_sys.PR_SET_NO_NEW_PRIVS, 1)
        if not seccomp.install():
            raise OSError("unsupported architecture")
        import socket
        try:
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except PermissionError:
            return
        raise OSError("socket() was not refused")
    f["seccomp"] = _try(filtered)
    return f


if __name__ == "__main__":
    json.dump(features(), sys.stdout, indent=1)
    print()
