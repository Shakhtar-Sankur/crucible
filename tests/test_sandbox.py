"""Every limit the sandbox claims is checked by running code that runs into it. Tests that
need an isolation layer the machine does not allow are skipped, and say which."""

import os
import threading

import pytest

from crucible.config import Limits
from crucible.sandbox import Zygote


@pytest.fixture(scope="module")
def zygote():
    z = Zygote(preload=["json", "math"])
    yield z
    z.close()


def need(z, *layers):
    missing = [l for l in layers if not z.features.get(l)]
    if missing:
        pytest.skip(f"this machine does not allow: {missing}")


def test_runs_code_and_keeps_state(zygote):
    with zygote.spawn() as sb:
        assert sb.exec("x = 6 * 7")["ok"]
        r = sb.exec("print(x); import sys; print('err', file=sys.stderr)")
        assert r["stdout"] == "42\n" and r["stderr"] == "err\n"


def test_errors_are_reported_not_fatal(zygote):
    with zygote.spawn() as sb:
        r = sb.exec("1 / 0")
        assert not r["ok"] and r["error"]["type"] == "ZeroDivisionError"
        r = sb.exec("raise SystemExit(3)")
        assert r["error"]["type"] == "SystemExit"
        assert sb.exec("print('still here')")["stdout"] == "still here\n"


def test_output_is_capped(zygote):
    with zygote.spawn(Limits(output_bytes=1000)) as sb:
        r = sb.exec("print('x' * 100000)")
        assert r["truncated"] and len(r["stdout"]) == 1000


def test_no_network(zygote):
    need(zygote, "ns_net")
    with zygote.spawn() as sb:
        r = sb.exec("import socket\n"
                    "s = socket.socket()\n"
                    "s.settimeout(2)\n"
                    "s.connect(('1.1.1.1', 53))")
        assert not r["ok"] and r["error"]["type"] in ("OSError", "TimeoutError", "ConnectionRefusedError")
        # its own network namespace: only a loopback interface, and it is down
        r = sb.exec("import socket; print(sorted(n for _, n in socket.if_nameindex()))")
        assert r["stdout"].strip() in ("['lo']", "[]")


def test_memory_limit(zygote):
    with zygote.spawn(Limits(memory_mb=256)) as sb:
        r = sb.exec("b = bytearray(1 << 30)")
        assert not r["ok"] and r["error"]["type"] == "MemoryError"
        assert sb.exec("print('alive')")["stdout"] == "alive\n"


def test_cpu_limit_ends_the_sandbox(zygote):
    with zygote.spawn(Limits(cpu_seconds=1)) as sb:
        r = sb.exec("while True: pass", timeout=20)
        assert not r["ok"] and r.get("died")


def test_wall_clock_timeout_kills_it(zygote):
    sb = zygote.spawn()
    pid = sb.pid
    r = sb.exec("import time; time.sleep(30)", timeout=1)
    assert r.get("timeout")
    for _ in range(100):
        if not os.path.exists(f"/proc/{pid}") or open(f"/proc/{pid}/stat").read().split()[2] == "Z":
            break
        threading.Event().wait(0.05)
    else:
        pytest.fail("the sandbox survived its timeout")


def test_process_limit(zygote):
    need(zygote, "setuid")
    with zygote.spawn(Limits(processes=8)) as sb:
        r = sb.exec("import os\n"
                    "made = 0\n"
                    "try:\n"
                    "    for _ in range(50):\n"
                    "        if os.fork() == 0:\n"
                    "            import time; time.sleep(5); os._exit(0)\n"
                    "        made += 1\n"
                    "except OSError as e:\n"
                    "    print('stopped after', made)\n", timeout=20)
        assert r["ok"] and r["stdout"].startswith("stopped after")
        assert int(r["stdout"].split()[-1]) < 8


def test_file_size_limit(zygote):
    with zygote.spawn(Limits(file_mb=1)) as sb:
        r = sb.exec("open('big', 'wb').write(b'x' * (2 << 20))")
        assert not r["ok"] and r["error"]["type"] == "OSError"


def test_cannot_write_outside_its_workspace(zygote):
    need(zygote, "setuid")
    with zygote.spawn() as sb:
        assert sb.exec("open('mine.txt', 'w').write('ok')")["ok"]
        for path in ("/etc/crucible-test", "/usr/crucible-test", "/root/crucible-test"):
            r = sb.exec(f"open({path!r}, 'w').write('no')")
            assert not r["ok"] and r["error"]["type"] in ("PermissionError", "OSError"), path


def test_sandboxes_do_not_see_each_other(zygote):
    need(zygote, "ns_mount", "ns_pid")
    with zygote.spawn() as a, zygote.spawn() as b:
        assert a.exec("open('secret.txt', 'w').write('a')")["ok"]
        r = b.exec("import os; print(os.listdir('.'), os.path.exists('secret.txt'))")
        assert "False" in r["stdout"]
        r = b.exec("import os; print(sorted(p for p in os.listdir('/proc') if p.isdigit()))")
        assert r["stdout"].strip() == "['1']"


def test_isolation_layers_reported(zygote):
    with zygote.spawn() as sb:
        for layer in ("limits", "no_new_privs"):
            assert sb.layers[layer]
        for name in ("mount", "net", "ipc", "uts", "pid"):
            assert sb.layers[f"ns_{name}"] == bool(zygote.features.get(f"ns_{name}"))


def test_many_at_once(zygote):
    out, errors = [], []

    def one(i):
        try:
            with zygote.spawn() as sb:
                out.append(sb.exec(f"print({i} * {i})")["stdout"])
        except Exception as e:
            errors.append(e)

    ts = [threading.Thread(target=one, args=(i,)) for i in range(32)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors and sorted(int(x) for x in out) == sorted(i * i for i in range(32))


def test_spawn_under_load_many_times(zygote):
    """The two startup messages may arrive in either order; spawning must not care."""
    errors = []

    def many():
        try:
            for _ in range(40):
                with zygote.spawn() as sb:
                    assert sb.exec("print(1)")["stdout"] == "1\n"
        except Exception as e:
            errors.append(repr(e))

    ts = [threading.Thread(target=many) for _ in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, errors[:3]
