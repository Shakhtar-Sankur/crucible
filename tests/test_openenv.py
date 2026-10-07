"""The coding environment through OpenEnv's own server (skipped without `pip install openenv`):
the HTTP endpoints, one-shot grading for a trainer, and a full episode over a WebSocket
session (run code, keep state, submit, get a reward)."""

import pytest

openenv = pytest.importorskip("openenv")
from fastapi.testclient import TestClient  # noqa: E402

from crucible.envs.server import build_app  # noqa: E402

ADD_OK = "def add(a, b):\n    return a + b\n"


@pytest.fixture(scope="module")
def client():
    with TestClient(build_app()) as c:
        yield c


def test_health_and_schema(client):
    assert client.get("/health").status_code == 200
    schema = client.get("/schema").json()
    text = str(schema)
    assert "code" in text and "tests_passed" in text


def test_one_shot_grade_over_http(client):
    r = client.post("/step", json={"action": {"code": ADD_OK, "kind": "submit", "task_id": "add"}}).json()
    assert r["done"] is True and r["reward"] == 1.0
    assert r["observation"]["tests_passed"] == 3
    r = client.post("/step", json={"action": {"code": "def add(a, b): return -999", "task_id": "add"}}).json()
    assert r["reward"] == pytest.approx(0.0)


def test_reset_over_http_returns_a_prompt(client):
    r = client.post("/reset", json={"seed": 3}).json()
    obs = r["observation"]
    assert obs["prompt"] and obs["tests_total"] > 0


def test_episode_over_websocket(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "data": {"task_id": "primes"}})
        obs = ws.receive_json()
        assert obs["type"] == "observation" and "primes_below" in obs["data"]["observation"]["prompt"]

        ws.send_json({"type": "step", "data": {"code": "scratch = [p for p in range(2, 20) if all(p % d for d in range(2, p))]", "kind": "run"}})
        r = ws.receive_json()["data"]
        assert r["done"] is False and r["observation"]["error"] is None

        ws.send_json({"type": "step", "data": {"code": "print(scratch)", "kind": "run"}})
        r = ws.receive_json()["data"]
        assert r["observation"]["stdout"] == "[2, 3, 5, 7, 11, 13, 17, 19]\n"   # state kept between runs

        sol = ("def primes_below(n):\n"
               "    return [p for p in range(2, n) if all(p % d for d in range(2, int(p ** 0.5) + 1))]\n")
        ws.send_json({"type": "step", "data": {"code": sol, "kind": "submit"}})
        r = ws.receive_json()["data"]
        assert r["done"] is True and r["reward"] == 1.0

        ws.send_json({"type": "state"})
        st = ws.receive_json()
        assert st["type"] == "state" and st["data"]["submitted"] is True and st["data"]["runs"] == 2


def test_hack_gets_nothing_over_the_wire(client):
    hack = "class A:\n    def __eq__(s, o): return True\ndef add(a, b): return A()"
    r = client.post("/step", json={"action": {"code": hack, "task_id": "add"}}).json()
    assert r["reward"] == 0.0
