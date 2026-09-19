"""Exact-wheel checks for Vertex Gemini hosted tools and their returned evidence."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path

import google.auth
from zhivex_ai import create_vertex, generate_grounded_text, generate_text, vertex_code_execution_tool, vertex_url_context_tool
from verify_vertex_integration import verify_wheel


async def run(args: argparse.Namespace) -> int:
    digest = verify_wheel(args.wheel)
    if args.output.exists():
        raise SystemExit("Use a fresh evidence path.")
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=args.project)
    provider = create_vertex(credentials=credentials, project_id=args.project, location="global")
    report = {
        "schema_version": 1, "provider": "vertex", "model": args.model,
        "mode": "standard-adc-hosted-tools", "location": "global",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "wheel_sha256": digest, "evidence_status": "integration-only",
        "operations": {}, "diagnostics": {},
    }

    async def search():
        result = await generate_grounded_text(
            model=provider.grounded_language_model(args.model),
            prompt="Use Google Search to find NASA's page about the total solar eclipse of August 12, 2026. Give a brief factual answer with sources.",
            max_tokens=2048, timeout_ms=60000, max_retries=0,
        )
        if not result.text or not result.sources or not result.queries or not result.supports:
            raise RuntimeError("Missing grounded text, sources, queries or citation supports")
        if not all(source.url.startswith("https://") for source in result.sources):
            raise RuntimeError("Invalid source URL")
        if not any(support.source_indices for support in result.supports):
            raise RuntimeError("Missing citation source associations")
        if any(index < 0 or index >= len(result.sources) for support in result.supports for index in support.source_indices):
            raise RuntimeError("Citation source index out of range")

    async def code():
        result = await generate_text(
            model=provider.native.language_model(args.model),
            prompt="Use the Python code execution tool to calculate sum(i*i for i in range(1, 101)). Execute the code and return the number.",
            tools={"python": vertex_code_execution_tool()},
            max_tokens=2048, timeout_ms=60000, max_retries=0,
        )
        parts = [part for message in result.messages for part in message.parts]
        if not any(getattr(part, "code", None) for part in parts):
            raise RuntimeError("Missing normalized generated code")
        if not any(getattr(part, "outcome", None) == "OUTCOME_OK" and "338350" in getattr(part, "output", "") for part in parts):
            raise RuntimeError("Missing successful normalized code result")

    async def url():
        result = await generate_text(
            model=provider.native.language_model(args.model),
            prompt="Read https://www.ietf.org/rfc/rfc9110.html with URL context and give its document title. Use the URL retrieval tool.",
            tools={"url": vertex_url_context_tool()},
            max_tokens=2048, timeout_ms=60000, max_retries=0,
        )
        candidates = result.steps[-1].response.raw_response.get("candidates", [])
        urls = [item for candidate in candidates for item in candidate.get("urlContextMetadata", {}).get("urlMetadata", [])]
        if not any(item.get("urlRetrievalStatus") == "URL_RETRIEVAL_STATUS_SUCCESS" for item in urls):
            raise RuntimeError("URL retrieval did not report success")
        if "HTTP Semantics" not in result.text:
            raise RuntimeError("Retrieved document title mismatch")

    for name, call in [("google-search-citations", search), ("code-execution", code), ("url-context", url)]:
        try:
            async with asyncio.timeout(90):
                await call()
            report["operations"][name] = "passed"
        except Exception as error:
            report["operations"][name] = "failed"
            report["diagnostics"][name] = {"error_type": type(error).__name__, "http_status": getattr(error, "status", None)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, report["operations"][name], flush=True)
    return int("failed" in report["operations"].values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--model", default="gemini-3.8-flash")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
