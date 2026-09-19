"""Exact-wheel native Vertex Responses text and SSE checks with synthetic input."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path

import google.auth
from zhivex_ai import create_vertex
from zhivex_ai.types import RetryOptions
from zhivex_ai._sse import parse_sse
from verify_vertex_integration import verify_wheel


async def run(args):
    digest = verify_wheel(args.wheel)
    if args.output.exists():
        raise SystemExit("Choose a fresh report path")
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=args.project)
    client = create_vertex(credentials=credentials, project_id=args.project, location=args.location).native.model_garden()
    report = {"schema_version": 1, "provider": "vertex", "model": args.model, "location": args.location, "mode": "standard-adc-native-responses", "wheel_sha256": digest, "recorded_at": datetime.now(timezone.utc).isoformat(), "evidence_status": "integration-only", "operations": {}, "diagnostics": {}}
    body = {"model": args.model, "input": "Reply exactly VERTEX_RESPONSES_OK.", "max_output_tokens": 512, "store": False}
    for streaming in (False, True):
        name = "streaming" if streaming else "generation"
        response = None
        lines = None
        try:
            async with asyncio.timeout(60):
                result = await client.responses({**body, "stream": streaming}, options=RetryOptions(timeout_ms=45000, max_retries=0))
                if streaming:
                    response = result
                    text = ""
                    completed = False
                    lines = response.iter_lines()
                    async for event in parse_sse(lines):
                        if event.data == "[DONE]":
                            continue
                        payload = json.loads(event.data)
                        if payload.get("type") == "response.output_text.delta":
                            text += payload["delta"]
                        if payload.get("type") == "response.completed":
                            completed = payload.get("response", {}).get("status") == "completed"
                        if payload.get("type") in {"error", "response.failed", "response.incomplete"}:
                            raise ValueError("Response stream failed or incomplete")
                else:
                    completed = result.get("status") == "completed"
                    text = "".join(part.get("text", "") for item in result.get("output", []) if item.get("type") == "message" for part in item.get("content", []) if part.get("type") == "output_text")
                if not completed or text.strip().rstrip(".") != "VERTEX_RESPONSES_OK":
                    raise ValueError("Incomplete response or marker mismatch")
                report["operations"][name] = "passed"
        except Exception as error:
            report["operations"][name] = "failed"
            report["diagnostics"][name] = {"error_type": type(error).__name__, "http_status": getattr(error, "status", None)}
        finally:
            if lines is not None and hasattr(lines, "aclose"):
                await lines.aclose()
            if response is not None and hasattr(response, "aclose"):
                await response.aclose()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, report["operations"][name], flush=True)
    return int(any(value != "passed" for value in report["operations"].values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="xai/grok-4.6")
    parser.add_argument("--location", default="global")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
