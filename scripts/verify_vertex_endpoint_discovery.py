"""Read-only exact-wheel endpoint/model-registry discovery; never inference certification."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path

import google.auth
from zhivex_ai import create_vertex
from zhivex_ai.types import RetryOptions
if __package__:
    from .verify_vertex_integration import verify_wheel
else:
    from verify_vertex_integration import verify_wheel


async def run(args):
    digest = verify_wheel(args.wheel)
    if args.output.exists():
        raise SystemExit("Choose a fresh evidence path.")
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=args.project)
    registry = args.registered_models
    collection = "models" if registry else "endpoints"
    count_key = "model_count" if registry else "endpoint_count"
    report = {"schema_version": 1, "provider": "vertex", "evidence_status": "integration-only",
              "scope": "model-registry-discovery-only" if registry else "endpoint-discovery-only", "wheel_sha256": digest,
              "recorded_at": datetime.now(timezone.utc).isoformat(), "locations": {}}
    for location in dict.fromkeys(args.location):
        item = {"list": "pending", count_key: 0, "deployed_model_count": 0, "pages": 0}
        report["locations"][location] = item
        client = create_vertex(credentials=credentials, project_id=args.project, location=location).native.model_garden()
        token = None
        first = None
        seen_tokens = set()
        seen_names = set()
        try:
            async with asyncio.timeout(90):
                for _ in range(5):
                    list_resources = client.list_registered_models if registry else client.list_endpoints
                    page = await list_resources(page_size=100, page_token=token, options=RetryOptions(timeout_ms=15000, max_retries=0))
                    endpoints = page.get(collection, [])
                    prefix = f"projects/{args.project}/locations/{location}/{collection}/"
                    for resource in endpoints:
                        name = resource.get("name", "")
                        if not name.startswith(prefix) or not name[len(prefix):] or "/" in name[len(prefix):]:
                            raise RuntimeError("Unexpected resource identity")
                        if name in seen_names:
                            raise RuntimeError("Duplicate resource in pagination")
                        seen_names.add(name)
                    item["pages"] += 1
                    item[count_key] += len(endpoints)
                    item["deployed_model_count"] += sum(len(e.get("deployedModels", [])) for e in endpoints)
                    if endpoints and first is None:
                        first = endpoints[0]["name"]
                    token = page.get("nextPageToken")
                    if not token:
                        break
                    if token in seen_tokens:
                        raise RuntimeError("Repeated page token")
                    seen_tokens.add(token)
                item["list"] = "incomplete-page-limit" if token else "passed"
                if first:
                    get_resource = client.get_registered_model if registry else client.get_endpoint
                    resource = await get_resource(first.rsplit("/", 1)[1], options=RetryOptions(timeout_ms=15000, max_retries=0))
                    item["get"] = "passed" if resource.get("name") == first else "failed"
                else:
                    item["get"] = "not-run-no-model" if registry else "not-run-no-endpoint"
        except Exception as error:
            item["list"] = "failed" if item["list"] == "pending" else item["list"]
            item["error"] = {"type": type(error).__name__, "http_status": getattr(error, "status", None)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(location, json.dumps(item), flush=True)
    return int(any(item["list"] != "passed" or "error" in item or item.get("get") == "failed" for item in report["locations"].values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", action="append", required=True)
    parser.add_argument("--registered-models", action="store_true", help="Read the project model registry instead of endpoints")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
