"""Serve the coding environment over OpenEnv's HTTP/WebSocket API.

    python -m crucible.envs.server --tasks data/mbpp/test.jsonl --port 8000

Endpoints (OpenEnv's create_app): POST /reset, POST /step, GET /state, GET /schema,
GET /health, WebSocket /ws for sessions."""

import argparse

from openenv.core.env_server import create_app

from ..tasks import BUILTIN, load_mbpp
from .coding import CodeAction, CodeObservation, CodeState, environment_class


def build_app(tasks=None, max_concurrent=64):
    env_cls = environment_class(tasks or BUILTIN)
    return create_app(env_cls, CodeAction, CodeObservation, env_name="crucible_coding",
                      max_concurrent_envs=max_concurrent, state_cls=CodeState)


def main(argv=None):
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=None, help="MBPP-style JSONL (default: the built-in tasks)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-concurrent", type=int, default=64)
    args = ap.parse_args(argv)
    tasks = load_mbpp(args.tasks) if args.tasks else BUILTIN
    uvicorn.run(build_app(tasks, args.max_concurrent), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
