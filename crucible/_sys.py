"""Thin ctypes wrappers for the Linux calls the sandbox needs (Python's os module lacks
mount and, before 3.12, unshare)."""

import ctypes
import errno
import os
import resource

_libc = ctypes.CDLL(None, use_errno=True)

CLONE_NEWNS = 0x00020000
CLONE_NEWUTS = 0x04000000
CLONE_NEWIPC = 0x08000000
CLONE_NEWUSER = 0x10000000
CLONE_NEWPID = 0x20000000
CLONE_NEWNET = 0x40000000

MS_RDONLY = 1
MS_NOSUID = 2
MS_NODEV = 4
MS_NOEXEC = 8
MS_REMOUNT = 32
MS_BIND = 4096
MS_MOVE = 8192
MS_REC = 16384
MS_PRIVATE = 1 << 18

PR_SET_NO_NEW_PRIVS = 38
PR_SET_PDEATHSIG = 1


def _check(ret, what):
    if ret != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"{what}: {os.strerror(e)}")


def unshare(flags):
    _check(_libc.unshare(ctypes.c_int(flags)), "unshare")


def mount(source, target, fstype, flags=0, data=None):
    _check(_libc.mount(source.encode() if source else None, target.encode(),
                       fstype.encode() if fstype else None, ctypes.c_ulong(flags),
                       data.encode() if data else None), f"mount {target}")


def prctl(option, arg2=0):
    _check(_libc.prctl(ctypes.c_int(option), ctypes.c_ulong(arg2), ctypes.c_ulong(0),
                       ctypes.c_ulong(0), ctypes.c_ulong(0)), "prctl")


def write_file(path, text):
    fd = os.open(path, os.O_WRONLY)
    try:
        os.write(fd, text.encode())
    finally:
        os.close(fd)


def set_limits(memory_bytes=None, cpu_seconds=None, processes=None, open_files=None, file_bytes=None):
    """Hard limits for this process and everything it starts."""
    def lim(kind, value):
        if value is not None:
            resource.setrlimit(kind, (value, value))
    lim(resource.RLIMIT_CORE, 0)
    lim(resource.RLIMIT_AS, memory_bytes)
    lim(resource.RLIMIT_CPU, cpu_seconds)
    lim(resource.RLIMIT_NPROC, processes)
    lim(resource.RLIMIT_NOFILE, open_files)
    lim(resource.RLIMIT_FSIZE, file_bytes)


def errno_name(e):
    return errno.errorcode.get(e, str(e))
