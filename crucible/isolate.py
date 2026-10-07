"""Turning a freshly forked process into a sandbox.

Layers, each used when the kernel allows it (probe.features() says which):
  - its own Linux user (uid_base + slot): file permissions keep it out of other sandboxes'
    files and of everything owned by root, and RLIMIT_NPROC counts only its own processes;
  - namespaces: user, mount, network (no interfaces up: no network at all), IPC, UTS and
    PID (it sees only its own processes, and killing its init kills all of them);
  - a private tmpfs over /tmp, /var/tmp and /dev/shm, size-limited (its workspace);
  - resource limits (memory, CPU time, processes, open files, file size, no core dumps);
  - no capabilities left, and no_new_privs so no setuid program can give any back.
Every step that fails because the environment forbids it is recorded, never ignored
silently: the sandbox reports exactly which layers are active."""

import ctypes
import os
import signal

from . import _sys

WORKDIR = "/tmp/work"
_CAP_HEADER_V3 = 0x20080522
_PR_CAPBSET_DROP = 24


def _drop_capabilities():
    for cap in range(64):
        try:
            _sys.prctl(_PR_CAPBSET_DROP, cap)
        except OSError:
            break
    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]
    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32),
                    ("inheritable", ctypes.c_uint32)]
    hdr, data = Header(_CAP_HEADER_V3, 0), (Data * 2)()
    if _sys._libc.capset(ctypes.byref(hdr), data) != 0:
        e = ctypes.get_errno()
        raise OSError(e, "capset")


def _private_tmpfs(size_mb):
    _sys.mount(None, "/", None, _sys.MS_REC | _sys.MS_PRIVATE)
    for d in ("/tmp", "/var/tmp", "/dev/shm"):
        if os.path.isdir(d):
            _sys.mount("tmpfs", d, "tmpfs", _sys.MS_NOSUID | _sys.MS_NODEV, f"size={size_mb}m,mode=1777")


def enter(limits, slot, features):
    """Called in the child right after the zygote's fork. Returns (layers, pid_namespace):
    the isolation actually in place. With a PID namespace the caller must fork once more
    (the next child is the namespace's init)."""
    layers = {}
    root = os.geteuid() == 0
    uid = features.get("uid_base", 200000) + slot if root and features.get("setuid") else os.geteuid()
    ns = 0
    for name, flag in (("mount", _sys.CLONE_NEWNS), ("net", _sys.CLONE_NEWNET), ("ipc", _sys.CLONE_NEWIPC),
                       ("uts", _sys.CLONE_NEWUTS), ("pid", _sys.CLONE_NEWPID)):
        if name == "net" and limits.network:
            continue
        if features.get(f"ns_{name}"):
            ns |= flag
    userns = bool(features.get("ns_user"))
    if userns:
        # Unprivileged first, then a user namespace in which it may set up the others.
        if root and uid != os.geteuid():
            os.setgroups([])
            os.setgid(uid)
            os.setuid(uid)
        _sys.unshare(_sys.CLONE_NEWUSER | ns)
        _sys.write_file("/proc/self/setgroups", "deny")
        _sys.write_file("/proc/self/uid_map", f"0 {uid} 1")
        _sys.write_file("/proc/self/gid_map", f"0 {uid} 1")
        layers["user"] = uid
        if ns & _sys.CLONE_NEWNS:
            _private_tmpfs(limits.tmpfs_mb)
    elif root:
        # No user namespaces: set up the rest as root, then drop to the sandbox's user.
        if ns:
            _sys.unshare(ns)
            if ns & _sys.CLONE_NEWNS:
                _private_tmpfs(limits.tmpfs_mb)
        if uid != os.geteuid():
            # dropped in finish(), after the PID namespace's init has mounted its /proc
            layers["pending_uid"] = uid
    else:
        ns = 0  # an unprivileged process without user namespaces can create none
    for name, flag in (("mount", _sys.CLONE_NEWNS), ("net", _sys.CLONE_NEWNET), ("ipc", _sys.CLONE_NEWIPC),
                       ("uts", _sys.CLONE_NEWUTS), ("pid", _sys.CLONE_NEWPID)):
        layers[f"ns_{name}"] = bool(ns & flag)
    layers["tmpfs"] = bool(ns & _sys.CLONE_NEWNS)
    # Its workspace: inside its private tmpfs, or else a directory only its user can enter.
    work = WORKDIR if layers["tmpfs"] else f"/tmp/crucible-{os.getpid()}"
    os.makedirs(work, mode=0o700, exist_ok=True)
    if "pending_uid" in layers:
        os.chown(work, layers["pending_uid"], layers["pending_uid"])
    layers["workdir"] = work
    return layers, bool(ns & _sys.CLONE_NEWPID)


def finish(limits, layers, as_pid1):
    """The last steps, in the process that will run the code (the PID namespace's init
    when there is one): its own /proc, limits, no capabilities, no new privileges."""
    if as_pid1 and layers.get("ns_mount"):
        try:
            _sys.mount("proc", "/proc", "proc", _sys.MS_NOSUID | _sys.MS_NODEV | _sys.MS_NOEXEC)
            layers["own_proc"] = True
        except OSError:
            layers["own_proc"] = False
    uid = layers.pop("pending_uid", None)
    if uid is not None:
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)
        layers["user"] = uid
    nproc = limits.processes if "user" in layers else None  # only meaningful for its own uid
    _sys.set_limits(memory_bytes=limits.memory_mb << 20, cpu_seconds=limits.cpu_seconds, processes=nproc,
                    open_files=limits.open_files, file_bytes=limits.file_mb << 20)
    layers["limits"] = True
    try:
        _drop_capabilities()
        layers["no_capabilities"] = True
    except OSError:
        layers["no_capabilities"] = False
    _sys.prctl(_sys.PR_SET_NO_NEW_PRIVS, 1)
    layers["no_new_privs"] = True
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    os.chdir(layers["workdir"])
    return layers
