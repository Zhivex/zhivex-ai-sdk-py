"""Read-only regional RAG corpus discovery; no corpus creation or retrieval."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path

import google.auth
from zhivex_ai import create_vertex
from zhivex_ai.types import RetryOptions
from verify_vertex_integration import verify_wheel


async def run(args):
    digest = verify_wheel(args.wheel)
    if args.output.exists():
        raise SystemExit("Choose a fresh evidence path.")
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=args.project)
    report = {"schema_version": 1, "provider": "vertex", "scope": "rag-corpus-discovery-only",
              "evidence_status": "integration-only", "wheel_sha256": digest,
              "recorded_at": datetime.now(timezone.utc).isoformat(), "locations": {}}
    for location in dict.fromkeys(args.location):
        item = {"list": "pending", "pages": 0, "corpus_count": 0}
        report["locations"][location] = item
        client = create_vertex(credentials=credentials, project_id=args.project, location=location).native.rag()
        token = None
        seen_tokens = set()
        try:
            async with asyncio.timeout(90):
                for _ in range(10):
                    page = await client.list(page_size=100, page_token=token, options=RetryOptions(timeout_ms=20000, max_retries=0))
                    entries = page.get("ragCorpora", [])
                    if not isinstance(entries, list):
                        raise ValueError("Invalid corpus collection")
                    item["pages"] += 1
                    item["corpus_count"] += len(entries)
                    token = page.get("nextPageToken")
                    if not token:
                        break
                    if not isinstance(token, str) or token in seen_tokens:
                        raise ValueError("Invalid or repeated pagination token")
                    seen_tokens.add(token)
                item["list"] = "incomplete-page-limit" if token else "passed"
        except Exception as error:
            item["list"] = "failed"
            item["error"] = {"type": type(error).__name__, "http_status": getattr(error, "status", None)}
        if args.engine_config:
            try:
                config = await client.get_engine_config(RetryOptions(timeout_ms=20000, max_retries=0))
                if not isinstance(config, dict):
                    raise ValueError("Invalid engine config")
                item["engine_config"] = "passed"
                managed = config.get("ragManagedDbConfig") or {}
                item["managed_db_tiers_present"] = [key for key in ("basic", "scaled", "unprovisioned") if key in managed]
            except Exception as error:
                item["engine_config"] = "failed"
                item["engine_config_error"] = {"type": type(error).__name__, "http_status": getattr(error, "status", None)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(location, json.dumps(item), flush=True)
    return int(any(item["list"] != "passed" or item.get("engine_config") == "failed" for item in report["locations"].values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--engine-config", action="store_true", help="Also read regional engine config; never updates it")
    parser.add_argument("--location", action="append", required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
