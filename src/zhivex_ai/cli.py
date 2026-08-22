from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import os
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .agent import Agent, run_agent
from .agent_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationExpectations,
    create_agent_evaluation_report,
    run_agent_evaluation,
)
from .protocols import A2AAgentExecutor, create_a2a_agent_card, create_a2a_app
from .responses_host import create_agent_playground_app, create_responses_app


def _sdk_version() -> str:
    try:
        return version("zhivex-ai-sdk")
    except PackageNotFoundError:
        return "0.19.0"


def load_agent(spec: str) -> Agent:
    """Load ``module:attribute`` and require an Agent instance.

    Loading a module executes that application's Python code. Only load trusted
    local modules, exactly as you would when starting an ASGI application.
    """

    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError('Agent references must use the form "module:attribute".')
    value: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        value = getattr(value, part)
    if callable(value) and not isinstance(value, Agent):
        value = value()
    if not isinstance(value, Agent):
        raise TypeError(f'"{spec}" did not resolve to a zhivex_ai.Agent.')
    return value


def _read_dataset(path: str) -> list[AgentEvaluationCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        raise ValueError("Evaluation datasets must be a JSON list or an object with a cases list.")
    cases: list[AgentEvaluationCase] = []
    names: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise ValueError("Every evaluation case requires a string name.")
        if not raw["name"]:
            raise ValueError("Evaluation case names cannot be empty.")
        if raw["name"] in names:
            raise ValueError(f'Duplicate evaluation case name "{raw["name"]}".')
        names.add(raw["name"])
        raw_expectations = raw.get("expectations")
        expectations = None
        if raw_expectations is not None:
            if not isinstance(raw_expectations, dict):
                raise ValueError("Evaluation expectations must be a JSON object.")
            expectations = AgentEvaluationExpectations(**raw_expectations)
        cases.append(
            AgentEvaluationCase(
                name=raw["name"],
                prompt=raw.get("prompt"),
                expectations=expectations,
                metadata=raw.get("metadata") or {},
            )
        )
    if not cases:
        raise ValueError("Evaluation datasets cannot be empty.")
    return cases


def _write_text_atomic(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(target)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


async def _run_command(args: argparse.Namespace) -> int:
    result = await run_agent(agent=load_agent(args.agent), prompt=args.prompt)
    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "agent": result.agent_name,
                    "text": result.text,
                    "finish_reason": result.finish_reason,
                },
                ensure_ascii=False,
            )
        )
    else:
        print(result.text)
    return 0


async def _eval_command(args: argparse.Namespace) -> int:
    if not math.isfinite(args.min_pass_rate) or not 0 <= args.min_pass_rate <= 1:
        raise ValueError("--min-pass-rate must be a finite number between 0 and 1.")
    if args.max_mean_latency_ms is not None and (
        not math.isfinite(args.max_mean_latency_ms) or args.max_mean_latency_ms < 0
    ):
        raise ValueError("--max-mean-latency-ms must be a finite non-negative number.")
    if args.output_json and args.output_junit:
        if Path(args.output_json).resolve() == Path(args.output_junit).resolve():
            raise ValueError("--output-json and --output-junit must use different paths.")

    dataset = _read_dataset(args.dataset)
    result = await run_agent_evaluation(
        agent=load_agent(args.agent),
        dataset=dataset,
        repetitions=args.repetitions,
        max_concurrency=args.max_concurrency,
    )
    report = create_agent_evaluation_report(result)
    if report.trial_pass_rate < args.min_pass_rate:
        report.gate_failures.append(
            f"Expected trial_pass_rate >= {args.min_pass_rate}, got {report.trial_pass_rate}."
        )
    mean_latency_ms = report.metrics.get("mean_latency_ms")
    if (
        mean_latency_ms is not None
        and args.max_mean_latency_ms is not None
        and mean_latency_ms > args.max_mean_latency_ms
    ):
        report.gate_failures.append(
            f"Expected mean_latency_ms <= {args.max_mean_latency_ms}, got {mean_latency_ms}."
        )
    if report.gate_failures:
        report.ok = False

    json_artifact = report.to_json()
    junit_artifact = report.to_junit_xml()
    if args.output_json:
        _write_text_atomic(args.output_json, f"{json_artifact}\n")
    if args.output_junit:
        _write_text_atomic(args.output_junit, f"{junit_artifact}\n")
    print(json_artifact)
    return 0 if report.ok else 1


def _inspect_command(args: argparse.Namespace) -> int:
    agent = load_agent(args.agent)
    model = agent.model
    tool_names = sorted(name for name, _ in agent.tools.items())
    print(
        json.dumps(
            {
                "name": agent.name,
                "provider": getattr(model, "provider", None),
                "model_id": getattr(model, "model_id", None),
                "tools": tool_names,
                "subagents": sorted(agent.subagents),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _serve_command(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError('Serving requires `pip install "zhivex-ai-sdk[api]"`.') from error
    agent = load_agent(args.agent)
    if args.protocol == "a2a":
        executor = A2AAgentExecutor(agent)
        base_url = (args.public_url or f"http://{args.host}:{args.port}").rstrip("/")
        card = create_a2a_agent_card(
            agent,
            url=f"{base_url}/a2a",
            version=args.agent_version,
        )
        app = create_a2a_app(executor=executor, card=card)
    else:
        app = create_responses_app(agents={args.model_alias: agent})
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _playground_command(args: argparse.Namespace) -> int:
    if args.host.strip().lower() not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError(
            "The development playground is restricted to a loopback host; "
            "use `zhivex serve` behind application-owned authentication for remote access."
        )
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError('The playground requires `pip install "zhivex-ai-sdk[api]"`.') from error
    app = create_agent_playground_app(agents={args.model_alias: load_agent(args.agent)})
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zhivex", description="Run, evaluate, inspect, and serve Zhivex agents.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_sdk_version()}")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect", help="Inspect a trusted local Agent definition.")
    inspect_parser.add_argument("agent", help="Python reference in module:attribute form.")

    run_parser = commands.add_parser("run", help="Run a trusted local Agent once.")
    run_parser.add_argument("agent", help="Python reference in module:attribute form.")
    run_parser.add_argument("--prompt", required=True)
    run_parser.add_argument("--json", action="store_true")

    eval_parser = commands.add_parser("eval", help="Run an evaluation JSON dataset.")
    eval_parser.add_argument("agent", help="Python reference in module:attribute form.")
    eval_parser.add_argument("--dataset", required=True)
    eval_parser.add_argument("--repetitions", type=int, default=1)
    eval_parser.add_argument("--max-concurrency", type=int, default=1)
    eval_parser.add_argument("--output-json")
    eval_parser.add_argument("--output-junit")
    eval_parser.add_argument("--min-pass-rate", type=float, default=1.0)
    eval_parser.add_argument("--max-mean-latency-ms", type=float)

    serve_parser = commands.add_parser("serve", help="Serve an Agent over a beta protocol adapter.")
    serve_parser.add_argument("agent", help="Python reference in module:attribute form.")
    serve_parser.add_argument("--protocol", choices=("responses", "a2a"), default="responses")
    serve_parser.add_argument("--model-alias", default="default")
    serve_parser.add_argument("--agent-version", default="0.1.0")
    serve_parser.add_argument("--public-url")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    playground_parser = commands.add_parser("playground", help="Start the local Responses playground.")
    playground_parser.add_argument("agent", help="Python reference in module:attribute form.")
    playground_parser.add_argument("--model-alias", default="default")
    playground_parser.add_argument("--host", default="127.0.0.1")
    playground_parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect_command(args)
        if args.command == "run":
            return asyncio.run(_run_command(args))
        if args.command == "eval":
            return asyncio.run(_eval_command(args))
        if args.command == "serve":
            return _serve_command(args)
        if args.command == "playground":
            return _playground_command(args)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"zhivex: {error}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command {args.command!r}.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
