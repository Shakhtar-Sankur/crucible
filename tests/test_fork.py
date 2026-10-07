"""M1: forking a live sandbox. A branch starts with its parent's Python state and files and
then lives its own life; killing one never touches the others."""

import os
import threading

import pytest

from crucible.config import Limits
from crucible.sandbox import Zygote


@pytest.fixture(scope="module")
def zygote():
    z = Zygote(preload=["json"])
    yield z
    z.close()


def test_branch_starts_from_the_parent_state_and_files(zygote):
    with zygote.spawn() as sb:
        sb.exec("data = list(range(1000)); open('notes.txt', 'w').write('v1')")
        br = sb.fork()
        try:
            r = br.exec("print(len(data), open('notes.txt').read())")
            assert r["stdout"] == "1000 v1\n"
        finally:
            br.close()


def test_branches_diverge_independently(zygote):
    with zygote.spawn() as sb:
        sb.exec("x = 0; open('f.txt', 'w').write('base')")
        branches = [sb.fork() for _ in range(4)]
        for i, br in enumerate(branches):
            br.exec(f"x = {i + 1}; open('f.txt', 'w').write('branch{i + 1}')")
        for i, br in enumerate(branches):
            assert br.exec("print(x, open('f.txt').read())")["stdout"] == f"{i + 1} branch{i + 1}\n"
            br.close()
        assert sb.exec("print(x, open('f.txt').read())")["stdout"] == "0 base\n"


def test_branch_of_a_branch(zygote):
    with zygote.spawn() as sb:
        sb.exec("path = ['root']")
        a = sb.fork()
        a.exec("path.append('a')")
        b = a.fork()
        b.exec("path.append('b')")
        assert b.exec("print(path)")["stdout"] == "['root', 'a', 'b']\n"
        assert a.exec("print(path)")["stdout"] == "['root', 'a']\n"
        assert sb.exec("print(path)")["stdout"] == "['root']\n"
        b.close(), a.close()


def test_killing_a_branch_leaves_the_rest(zygote):
    with zygote.spawn() as sb:
        sb.exec("y = 7")
        br, other = sb.fork(), sb.fork()
        r = br.exec("import time; time.sleep(30)", timeout=1)
        assert r.get("timeout")
        assert other.exec("print(y)")["stdout"] == "7\n"
        assert sb.exec("print(y)")["stdout"] == "7\n"
        other.close()


def test_branch_keeps_the_sandbox_limits(zygote):
    with zygote.spawn(Limits(memory_mb=256)) as sb:
        br = sb.fork()
        r = br.exec("b = bytearray(1 << 30)")
        assert r["error"]["type"] == "MemoryError"
        if zygote.features.get("seccomp"):
            r = br.exec("import socket; socket.socket(socket.AF_INET)")
            assert r["error"]["type"] == "PermissionError"
        br.close()


def test_branch_pid_is_one_the_client_can_signal(zygote):
    with zygote.spawn() as sb:
        br = sb.fork()
        assert os.path.exists(f"/proc/{br.pid}")
        inner = br.ping()["pid"]
        if zygote.features.get("ns_pid"):
            assert inner != br.pid                  # translated out of the PID namespace
        br.close()


def test_many_branches_at_once(zygote):
    with zygote.spawn() as sb:
        sb.exec("base = 41")
        out, errs = [], []

        def one(i):
            try:
                br = sb_fork()
                out.append(br.exec(f"print(base + {i})")["stdout"])
                br.close()
            except Exception as e:
                errs.append(repr(e))

        lock = threading.Lock()

        def sb_fork():
            with lock:                                # one request at a time on the parent's socket
                return sb.fork()

        ts = [threading.Thread(target=one, args=(i,)) for i in range(16)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert not errs, errs[:3]
        assert sorted(int(x) for x in out) == [41 + i for i in range(16)]
