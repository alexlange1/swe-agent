#!/usr/bin/env python3
"""
Tau Mining Agent — CLI entry point for Bittensor subnet 66.

Invoked by the tau validator (via cli.js shim) as:
  node $TAU_AGENT_DIR/packages/coding-agent/dist/cli.js \
    --mode json --no-session --provider docker-proxy --model docker-proxy-model \
    -p "$PROMPT"

Output: pi-json line-delimited events on stdout.
Exit code 0 = completed (diff is scored), non-zero = error (round forfeited).
"""

import argparse
import json
import os
import sys
import uuid

from solver import TauSolver


def emit(event: dict) -> None:
    print(json.dumps(event), flush=True)


def emit_session() -> str:
    sid = str(uuid.uuid4())
    emit({"type": "session", "id": sid})
    return sid


def emit_tool_start() -> None:
    emit({"type": "tool_execution_start"})


def emit_turn_end(text: str) -> None:
    emit({
        "type": "turn_end",
        "message": {"content": [{"type": "text", "text": text}]},
    })


class EventEmittingSolver(TauSolver):
    """Thin subclass that emits tool_execution_start for validator metrics."""

    def _run_tools(self, response, edit_errors):
        results, edit_made, read_count, new_errors = super()._run_tools(
            response, edit_errors
        )
        tool_count = sum(
            1 for block in response.content if block.type == "tool_use"
        )
        for _ in range(tool_count):
            emit_tool_start()
        return results, edit_made, read_count, new_errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Tau competitive mining agent")
    parser.add_argument("--mode", default="interactive")
    parser.add_argument("--no-session", action="store_true")
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()

    repo_path = (
        args.repo
        or os.environ.get("TAU_REPO_DIR")
        or os.environ.get("PI_REPO_DIR", "/work/repo")
    )
    model = "docker-proxy-model"

    if args.mode == "json":
        emit_session()

    solver = EventEmittingSolver(repo_path=repo_path, model=model)

    try:
        result_text, success = solver.solve(args.prompt)
    except Exception as e:
        result_text = f"Agent error: {type(e).__name__}: {e}"
        success = False

    if args.mode == "json":
        emit_turn_end(result_text if result_text else "Done.")
    else:
        print(result_text)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
