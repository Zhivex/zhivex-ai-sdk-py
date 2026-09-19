"""Exact-wheel Memory Bank generation, retrieval and scope-isolation check."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import google.auth
from zhivex_ai import create_vertex
from zhivex_ai.types import RetryOptions
from verify_vertex_integration import verify_wheel


async def run(args: argparse.Namespace) -> int:
    digest = verify_wheel(args.wheel)
    if args.output.exists() or args.state.exists():
        raise SystemExit("Use fresh output/checkpoint paths; reconcile existing resources first.")
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=args.project)
    platform = create_vertex(credentials=credentials, project_id=args.project, location=args.location).native.agent_platform()
    options = RetryOptions(timeout_ms=45000, max_retries=0)
    report = {
        "schema_version": 1, "provider": "vertex", "mode": "standard-adc-memory",
        "location": args.location, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "wheel_sha256": digest, "evidence_status": "integration-only",
        "operations": {}, "diagnostics": {},
    }
    agent_name = None

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    async def completed(operation):
        if operation.get("name") and not operation.get("done"):
            operation = await platform.wait_operation(operation["name"], timeout_ms=120000)
        if operation.get("error"):
            raise RuntimeError("Cloud operation terminated with an error")
        return operation.get("response", {})

    async def check(name, call):
        try:
            async with asyncio.timeout(150):
                result = await call()
            report["operations"][name] = "passed"
            return result
        except Exception as error:
            report["operations"][name] = "failed"
            report["diagnostics"][name] = {"error_type": type(error).__name__, "http_status": getattr(error, "status", None)}
            raise
        finally:
            save()
            print(name, report["operations"][name], flush=True)

    try:
        async def create():
            nonlocal agent_name
            operation = await platform.agents().create({"displayName": "zhivex-memory-smoke-" + uuid.uuid4().hex[:8]}, options=options)
            name = operation.get("name", "")
            if "/reasoningEngines/" in name and "/operations/" in name:
                agent_name = name.rsplit("/operations/", 1)[0]
                with args.state.open("x") as file:
                    file.write(json.dumps({"agent_name": agent_name, "location": args.location}))
                args.state.chmod(0o600)
            agent = await completed(operation)
            agent_name = agent["name"]

        await check("agent-create", create)
        bank = platform.memory_bank(agent_name)
        scope = {"user_id": "synthetic-a"}

        async def generate():
            await completed(await bank.generate({
                "scope": scope,
                "directContentsSource": {"events": [{"content": {
                    "role": "user", "parts": [{"text": "My favorite color is turquoise. I always prefer turquoise for my notebook covers. Please remember this preference."}],
                }}]},
            }, options))

        await check("memory-generate", generate)

        async def retrieve(isolated=False):
            result = await bank.retrieve({"scope": {"user_id": "synthetic-b"} if isolated else scope, "similaritySearchParams": {"searchQuery": "favorite color for notebook covers"}}, options)
            memories = result.get("retrievedMemories", [])
            if isolated:
                if memories:
                    raise RuntimeError("Memory crossed the requested scope")
            elif "turquoise" not in json.dumps(memories).lower():
                report["diagnostics"]["retrieve_response_keys"] = sorted(result.keys())
                raise RuntimeError("Generated preference not retrieved")

        await check("generated-memory-retrieve", retrieve)
        await check("scope-isolation", lambda: retrieve(True))
    except Exception:
        pass
    finally:
        if agent_name:
            async def cleanup():
                await completed(await platform.agents().delete(agent_name, force=True, options=options))
                args.state.write_text(json.dumps({"cleanup": "complete"}))
            try:
                await check("agent-delete", cleanup)
            except Exception:
                pass
    return int(any(value != "passed" for value in report["operations"].values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="global")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
