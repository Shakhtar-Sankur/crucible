"""Limits for one sandbox."""

from dataclasses import asdict, dataclass


@dataclass
class Limits:
    memory_mb: int = 512        # address space (RLIMIT_AS)
    cpu_seconds: int = 20       # CPU time (RLIMIT_CPU); wall-clock limits are per call
    processes: int = 64         # processes and threads of the sandbox's user (RLIMIT_NPROC)
    open_files: int = 256
    file_mb: int = 64           # largest file it may write (RLIMIT_FSIZE)
    tmpfs_mb: int = 128         # its private /tmp (the workspace lives there)
    output_bytes: int = 1 << 20 # stdout/stderr kept per call
    network: bool = False       # False: a network namespace with no interfaces up

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
