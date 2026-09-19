"""Resumable exact-wheel RAG corpus lifecycle using a temporary empty corpus."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import google.auth
from zhivex_ai import create_vertex
from zhivex_ai.types import RetryOptions
try:
    from verify_vertex_integration import verify_wheel
except ModuleNotFoundError:
    from scripts.verify_vertex_integration import verify_wheel


async def run(args):
    digest = verify_wheel(args.wheel)
    if args.output.exists():
        raise SystemExit("Choose a fresh report path.")
    identity = {"project": args.project, "location": args.location, "wheel_sha256": digest}
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state.get("identity") != identity:
            raise SystemExit("Checkpoint target or wheel differs.")
        if not state.get("create_operation"):
            raise SystemExit("Unknown creation outcome; reconcile before another submission.")
    else:
        state = {"identity": identity, "display_name": "zhivex-rag-smoke-" + uuid4().hex[:12]}
        with args.state.open("x") as handle:
            handle.write(json.dumps(state))
        args.state.chmod(0o600)
    def save():
        args.state.write_text(json.dumps(state))
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=args.project)
    rag = create_vertex(credentials=credentials, project_id=args.project, location=args.location).native.rag()
    options = RetryOptions(timeout_ms=20000, max_retries=0)
    report = {"schema_version": 1, "provider": "vertex", "scope": "empty-rag-corpus-lifecycle",
              "evidence_status": "integration-only", "wheel_sha256": digest, "location": args.location,
              "recorded_at": datetime.now(timezone.utc).isoformat(), "operations": {}, "diagnostics": {}}
    stage = "create"
    try:
        if not state.get("create_operation"):
            operation = await rag.create({"displayName": state["display_name"]}, options=options)
            state["create_operation"] = operation["name"]
            save()
        stage = "create-poll"
        if not state.get("corpus"):
            operation = await rag.wait_operation(state["create_operation"], poll_interval_ms=5000, timeout_ms=120000)
            if operation.get("error"):
                report["diagnostics"]["operation_error_code"] = operation["error"].get("code")
                raise RuntimeError("Create operation failed")
            state["corpus"] = operation["response"]["name"]
            save()
        report["operations"]["create"] = "passed"
        print("create passed; resource checkpoint saved", flush=True)
        if not state.get("delete_operation") and not state.get("deleted"):
            stage = "get"
            corpus = await rag.get(state["corpus"], options)
            if corpus.get("displayName") != state["display_name"]:
                raise RuntimeError("Corpus identity mismatch")
            report["operations"][stage] = "passed"
            state["get_passed"] = True
            save()
            stage = "update"
            if not state.get("update_operation"):
                operation = await rag.update(state["corpus"], {"name": state["corpus"], "description": "Temporary SDK lifecycle check"}, options=options)
                state["update_operation"] = operation["name"]
                save()
            operation = await rag.wait_operation(state["update_operation"], poll_interval_ms=5000, timeout_ms=120000)
            if operation.get("error"):
                raise RuntimeError("Update operation failed")
            updated = await rag.get(state["corpus"], options)
            if updated.get("description") != "Temporary SDK lifecycle check":
                raise RuntimeError("Updated description mismatch")
            state["update_passed"] = True
            report["operations"][stage] = "passed"
            save()
    except Exception as error:
        report["operations"][stage] = "pending" if isinstance(error, TimeoutError) else "failed"
        report["diagnostics"][stage] = {"error_type": type(error).__name__, "http_status": getattr(error, "status", None)}
        try:
            upstream = json.loads(getattr(error, "response_body", None) or "{}").get("error", {})
            message = str(upstream.get("message", "")).lower()
            report["diagnostics"][stage]["error_categories"] = [word for word in ("embedding", "model", "quota", "allowlist", "allowlisted", "region", "configuration", "managed", "required") if word in message]
            report["diagnostics"][stage]["upstream_status"] = upstream.get("status")
        except (ValueError, TypeError, AttributeError):
            pass
    # Clean up only the exact corpus captured from our creation operation.
    if state.get("corpus") and not state.get("deleted"):
        stage = "delete"
        try:
            if not state.get("delete_operation"):
                operation = await rag.delete(state["corpus"], options=options)
                state["delete_operation"] = operation["name"]
                save()
            stage = "delete-poll"
            operation = await rag.wait_operation(state["delete_operation"], poll_interval_ms=5000, timeout_ms=120000)
            if operation.get("error"):
                raise RuntimeError("Delete operation failed")
            state["deleted"] = True
            save()
        except Exception as error:
            report["operations"][stage] = "pending" if isinstance(error, TimeoutError) else "failed"
            report["diagnostics"][stage] = {"error_type": type(error).__name__, "http_status": getattr(error, "status", None)}
    if state.get("deleted"):
        report["operations"]["delete"] = "passed"
        if state.get("get_passed"):
            report["operations"]["get"] = "passed"
    if state.get("update_passed"):
        report["operations"]["update"] = "passed"
    report["cleanup_complete"] = bool(state.get("deleted"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["operations"]), flush=True)
    return int(not report["cleanup_complete"] or any(report["operations"].get(key) != "passed" for key in ("create", "get", "update", "delete")) or any(value != "passed" for value in report["operations"].values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
