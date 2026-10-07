"""A seccomp-BPF filter: the kernel refuses chosen system calls for the sandbox and
everything it starts. The second line of defence behind the namespaces, and the first
where namespaces are not allowed (Kaggle's containers, for one).

Refused (EPERM / EACCES, the sandbox's code sees an ordinary error):
  - network: socket() for anything but AF_UNIX (no network without a network namespace);
  - acting on other processes: ptrace, process_vm_readv/writev, kcmp, pidfd_getfd;
  - changing the system or its own confinement: mount, umount2, pivot_root, chroot, unshare,
    setns, swapon/off, reboot, kexec, (f)init/delete_module, bpf, perf_event_open, keyctl,
    add_key, request_key, userfaultfd, open_by_handle_at, iopl, ioperm, io_uring_setup;
  - the x32 ABI and any other architecture (killed: a way around a filter by syscall number).
x86_64 only; elsewhere install() reports False and the sandbox says so."""

import ctypes
import errno
import platform

from . import _sys

_AUDIT_ARCH_X86_64 = 0xC000003E
_X32_BIT = 0x40000000
_LD_W_ABS, _JEQ_K, _JGE_K, _RET_K = 0x20, 0x15, 0x35, 0x06
_RET_ALLOW, _RET_ERRNO, _RET_KILL = 0x7FFF0000, 0x00050000, 0x80000000
_PR_SET_SECCOMP, _MODE_FILTER = 22, 2
_AF_UNIX = 1

# x86_64 system call numbers
_SOCKET = 41
_DENY = {
    "ptrace": 101, "process_vm_readv": 310, "process_vm_writev": 311, "kcmp": 312, "pidfd_getfd": 438,
    "mount": 165, "umount2": 166, "pivot_root": 155, "chroot": 161, "unshare": 272, "setns": 308,
    "swapon": 167, "swapoff": 168, "reboot": 169, "kexec_load": 246, "kexec_file_load": 320,
    "init_module": 175, "finit_module": 313, "delete_module": 176, "bpf": 321, "perf_event_open": 298,
    "keyctl": 250, "add_key": 248, "request_key": 249, "userfaultfd": 323, "open_by_handle_at": 304,
    "iopl": 172, "ioperm": 173, "io_uring_setup": 425, "fsopen": 430, "fsmount": 432, "move_mount": 429,
    "open_tree": 428, "mount_setattr": 442,
}


class _Insn(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8), ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class _Prog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_Insn))]


def program(allow_network=False):
    """The filter as (code, jt, jf, k) tuples. seccomp_data: nr at 0, arch at 4, args from 16."""
    p = [(_LD_W_ABS, 0, 0, 4),                      # arch
         (_JEQ_K, 1, 0, _AUDIT_ARCH_X86_64),
         (_RET_K, 0, 0, _RET_KILL),
         (_LD_W_ABS, 0, 0, 0),                      # system call number
         (_JGE_K, 0, 1, _X32_BIT),
         (_RET_K, 0, 0, _RET_KILL)]
    for nr in sorted(_DENY.values()):
        p += [(_JEQ_K, 0, 1, nr), (_RET_K, 0, 0, _RET_ERRNO | errno.EPERM)]
    if not allow_network:
        p += [(_JEQ_K, 0, 4, _SOCKET),
              (_LD_W_ABS, 0, 0, 16),                # socket(domain, ...): domain
              (_JEQ_K, 0, 1, _AF_UNIX),
              (_RET_K, 0, 0, _RET_ALLOW),
              (_RET_K, 0, 0, _RET_ERRNO | errno.EACCES)]
    p.append((_RET_K, 0, 0, _RET_ALLOW))
    return p


def supported():
    return platform.machine() in ("x86_64", "AMD64")


def install(allow_network=False):
    """Installs the filter on this process (needs no_new_privs first). Irreversible."""
    if not supported():
        return False
    insns = program(allow_network)
    arr = (_Insn * len(insns))(*[_Insn(*i) for i in insns])
    prog = _Prog(len(insns), ctypes.cast(arr, ctypes.POINTER(_Insn)))
    if _sys._libc.prctl(_PR_SET_SECCOMP, _MODE_FILTER, ctypes.byref(prog), 0, 0) != 0:
        e = ctypes.get_errno()
        raise OSError(e, "seccomp")
    return True
