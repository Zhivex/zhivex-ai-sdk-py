"""Bounded read-only publisher catalog inventory from an exact installed wheel."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path

import google.auth
from zhivex_ai import create_vertex
from zhivex_ai.types import RetryOptions
try:
    from verify_vertex_integration import verify_wheel
except ModuleNotFoundError:
    from scripts.verify_vertex_integration import verify_wheel


async def inventory(client, publisher: str, max_pages: int = 20):
    models = {}
    seen_tokens = set()
    token = None
    for page_number in range(1, max_pages + 1):
        page = await client.list_models(publisher=publisher, page_size=100, page_token=token,
                                       options=RetryOptions(timeout_ms=20000, max_retries=0))
        entries = page.get("publisherModels", [])
        if not isinstance(entries, list):
            raise ValueError("Invalid publisher model collection")
        for entry in entries:
            name = entry.get("name") if isinstance(entry, dict) else None
            if not isinstance(name, str) or not name.startswith(f"publishers/{publisher}/models/"):
                raise ValueError("Invalid publisher model identity")
            # Public catalog metadata only; never retain page tokens or auth data.
            models[name] = {key: entry[key] for key in ("name", "versionId", "launchStage") if key in entry}
        token = page.get("nextPageToken")
        if not token:
            return {"status": "passed", "pages": page_number, "models": sorted(models.values(), key=lambda item: item["name"])}
        if not isinstance(token, str) or token in seen_tokens:
            raise ValueError("Invalid or repeated pagination token")
        seen_tokens.add(token)
    return {"status": "incomplete-page-limit", "pages": max_pages, "models": sorted(models.values(), key=lambda item: item["name"])}


async def run(args):
    digest = verify_wheel(args.wheel)
    if args.output.exists():
        raise SystemExit("Choose a fresh evidence path.")
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=args.project)
    client = create_vertex(credentials=credentials, project_id=args.project, location="global").native.model_garden()
    report = {"schema_version": 1, "provider": "vertex", "scope": "publisher-catalog-discovery-only",
              "evidence_status": "integration-only", "wheel_sha256": digest,
              "recorded_at": datetime.now(timezone.utc).isoformat(), "publishers": {}}
    for publisher in dict.fromkeys(args.publisher):
        try:
            async with asyncio.timeout(120):
                item = await inventory(client, publisher)
        except Exception as error:
            item = {"status": "failed", "error_type": type(error).__name__, "http_status": getattr(error, "status", None)}
        report["publishers"][publisher] = item
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(publisher, item["status"], len(item.get("models", [])), flush=True)
    return int(any(item["status"] != "passed" for item in report["publishers"].values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--publisher", action="append", required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
