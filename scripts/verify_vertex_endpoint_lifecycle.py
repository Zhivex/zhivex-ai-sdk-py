"""Resumable empty endpoint lifecycle; never deploys models or accelerators."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import asyncio
import json
from pathlib import Path
from uuid import uuid4

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
        raise SystemExit("Choose a fresh report path.")
    identity = {"project": args.project, "location": args.location, "wheel_sha256": digest}
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state.get("identity") != identity or not state.get("create_operation"):
            raise SystemExit("Checkpoint mismatch or uncertain creation; do not resubmit.")
    else:
        state = {"identity": identity, "endpoint_id": "zhivex-check-" + uuid4().hex[:16]}
        with args.state.open("x") as handle:
            handle.write(json.dumps(state))
        args.state.chmod(0o600)

    def save():
        args.state.write_text(json.dumps(state))

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=args.project)
    client = create_vertex(credentials=credentials, project_id=args.project, location=args.location).native.model_garden()
    options = RetryOptions(timeout_ms=20000, max_retries=0)
    report = {"schema_version": 1, "provider": "vertex", "scope": "empty-endpoint-lifecycle",
              "wheel_sha256": digest, "location": args.location, "evidence_status": "integration-only",
              "recorded_at": datetime.now(timezone.utc).isoformat(), "operations": {}, "diagnostics": {}}
    stage = "create"
    try:
        if not state.get("create_operation"):
            operation = await client.create_endpoint({"displayName": state["endpoint_id"]}, endpoint_id=state["endpoint_id"], options=options)
            state["create_operation"] = operation["name"]
            save()
        stage = "create-poll"
        if not state.get("created"):
            operation = await client.wait_operation(state["create_operation"], poll_interval_ms=3000, timeout_ms=60000)
            if operation.get("error"):
                raise RuntimeError("Endpoint creation operation failed")
            if operation.get("response", {}).get("name", "").rsplit("/", 1)[-1] != state["endpoint_id"]:
                raise RuntimeError("Created resource identity mismatch")
            state["created"] = True
            save()
        if not state.get("delete_submitted"):
            stage = "get"
            state["safe_to_delete"] = False
            save()
            endpoint = await client.get_endpoint(state["endpoint_id"], options=options)
            if endpoint.get("displayName") != state["endpoint_id"] or endpoint.get("deployedModels"):
                raise RuntimeError("Endpoint ownership or emptiness mismatch")
            state["get"] = True
            state["safe_to_delete"] = True
            save()
            stage = "update"
            await client.update_endpoint(state["endpoint_id"], {"description": "Temporary SDK endpoint check"}, update_mask="description", options=options)
            updated = await client.get_endpoint(state["endpoint_id"], options=options)
            if updated.get("description") != "Temporary SDK endpoint check":
                raise RuntimeError("Endpoint update mismatch")
            state["update"] = True
            save()
    except Exception as error:
        report["operations"][stage] = "pending" if isinstance(error, TimeoutError) else "failed"
        report["diagnostics"][stage] = {"error_type": type(error).__name__, "http_status": getattr(error, "status", None)}

    if state.get("safe_to_delete") and not state.get("deleted"):
        stage = "delete"
        try:
            if not state.get("delete_operation"):
                if state.get("delete_submitted"):
                    raise RuntimeError("Uncertain delete submission; reconcile before retry")
                state["delete_submitted"] = True
                save()
                operation = await client.delete_endpoint(state["endpoint_id"], options=options)
                state["delete_operation"] = operation["name"]
                save()
            stage = "delete-poll"
            operation = await client.wait_operation(state["delete_operation"], poll_interval_ms=3000, timeout_ms=60000)
            if operation.get("error"):
                raise RuntimeError("Endpoint deletion operation failed")
            stage = "absence"
            try:
                await client.get_endpoint(state["endpoint_id"], options=options)
            except Exception as error:
                if getattr(error, "status", None) != 404:
                    raise
            else:
                raise RuntimeError("Deleted endpoint is still readable")
            state["deleted"] = True
            save()
        except Exception as error:
            report["operations"][stage] = "pending" if isinstance(error, TimeoutError) else "failed"
            report["diagnostics"][stage] = {"error_type": type(error).__name__, "http_status": getattr(error, "status", None)}
    for operation, key in (("create", "created"), ("get", "get"), ("update", "update"), ("delete-and-absence", "deleted")):
        if state.get(key):
            report["operations"][operation] = "passed"
    report["cleanup_complete"] = bool(state.get("deleted"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["operations"]), flush=True)
    return int(not all(state.get(key) for key in ("created", "get", "update", "deleted")) or any(value != "passed" for value in report["operations"].values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("wheel", "state", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="us-central1")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
