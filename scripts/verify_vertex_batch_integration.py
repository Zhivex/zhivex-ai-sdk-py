"""Resumable one-row Vertex batch check; only a newly owned bucket is touched."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import google.auth
from google.auth.transport.requests import AuthorizedSession
from zhivex_ai import create_vertex
from zhivex_ai.types import RetryOptions

MARKER = "VERTEX_BATCH_OK"
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED"}


def validate_rows(rows: list[dict]) -> None:
    if len(rows) != 1 or rows[0].get("status") not in (None, ""):
        raise ValueError("Expected exactly one successful row")
    candidates = rows[0].get("response", {}).get("candidates", [])
    if len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
        raise ValueError("Missing complete candidate")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
    if text.strip() != MARKER:
        raise ValueError("Batch marker mismatch")


async def run(args):
    from verify_vertex_integration import verify_wheel

    digest = verify_wheel(args.wheel)
    if args.output.exists():
        raise SystemExit("Choose a fresh report path")
    identity = {"project": args.project, "model": args.model, "location": "global", "wheel_sha256": digest}
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state["identity"] != identity or state.get("cleaned"):
            raise SystemExit("Checkpoint identity mismatch or already cleaned")
        if not state.get("job"):
            raise SystemExit("Interrupted setup: reconcile resources before any further creation")
    else:
        state = {"identity": identity, "bucket": "zhivex-batch-" + uuid4().hex, "phase": "reserved"}
        fd = os.open(args.state, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as file:
            json.dump(state, file)

    def checkpoint():
        args.state.write_text(json.dumps(state))

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=args.project)
    session = AuthorizedSession(credentials)
    client = create_vertex(credentials=credentials, project_id=args.project, location="global").batches()
    bucket = state["bucket"]
    base = "https://storage.googleapis.com/storage/v1/b/" + bucket
    report = {"schema_version": 1, "provider": "vertex", "model": args.model, "location": "global", "mode": "standard-adc-batch", "wheel_sha256": digest, "recorded_at": datetime.now(timezone.utc).isoformat(), "evidence_status": "integration-only", "operations": {}, "diagnostics": {}}
    stage = "storage-setup"

    async def storage(method, url, **kwargs):
        response = await asyncio.to_thread(session.request, method, url, timeout=30, **kwargs)
        response.raise_for_status()
        return response

    async def objects():
        result = []
        token = None
        while True:
            params = {"pageToken": token} if token else {}
            page = (await storage("GET", base + "/o", params=params)).json()
            result.extend(page.get("items", []))
            token = page.get("nextPageToken")
            if not token:
                return result

    try:
        if not state.get("job"):
            state["phase"] = "bucket-dispatch"
            checkpoint()
            await storage("POST", "https://storage.googleapis.com/storage/v1/b", params={"project": args.project}, json={"name": bucket, "location": "US-CENTRAL1", "iamConfiguration": {"uniformBucketLevelAccess": {"enabled": True}, "publicAccessPrevention": "enforced"}, "softDeletePolicy": {"retentionDurationSeconds": "0"}, "labels": {"purpose": "zhivex-integration"}})
            state["bucket_created"] = True
            checkpoint()
            request = {"request": {"contents": [{"role": "user", "parts": [{"text": f"Reply exactly {MARKER} and nothing else."}]}], "generationConfig": {"maxOutputTokens": 128}}}
            await storage("POST", "https://storage.googleapis.com/upload/storage/v1/b/" + bucket + "/o", params={"uploadType": "media", "name": "input.jsonl", "ifGenerationMatch": "0"}, data=(json.dumps(request) + "\n").encode(), headers={"Content-Type": "application/jsonl"})
            report["operations"][stage] = "passed"
            stage = "create"
            state["phase"] = "job-dispatch"
            checkpoint()
            job = await client.create({"displayName": bucket, "model": "publishers/google/models/" + args.model, "inputConfig": {"instancesFormat": "jsonl", "gcsSource": {"uris": [f"gs://{bucket}/input.jsonl"]}}, "outputConfig": {"predictionsFormat": "jsonl", "gcsDestination": {"outputUriPrefix": f"gs://{bucket}/output/"}}}, RetryOptions(timeout_ms=60000, max_retries=0))
            if not job.get("name"):
                raise ValueError("Missing job identifier")
            state["job"] = job["name"]
            state["phase"] = "poll"
            checkpoint()
        report["operations"]["create"] = "passed"
        print("Batch job saved; polling the same job", flush=True)
        stage = "poll"
        job = await client.wait(state["job"], poll_interval_ms=10000, timeout_ms=120000)
        state["terminal_state"] = job["state"]
        checkpoint()
        report["diagnostics"]["job_state"] = job["state"]
        if job["state"] != "JOB_STATE_SUCCEEDED":
            report["diagnostics"]["job_error_code"] = job.get("error", {}).get("code")
            raise ValueError("Batch did not succeed")
        report["operations"][stage] = "passed"
        stage = "results"
        rows = []
        for item in await objects():
            if item["name"].startswith("output/") and item["name"].endswith(".jsonl"):
                response = await storage("GET", base + "/o/" + quote(item["name"], safe=""), params={"alt": "media", "generation": item["generation"]})
                rows.extend(json.loads(line) for line in response.text.splitlines() if line.strip())
        validate_rows(rows)
        report["operations"][stage] = "passed"
    except Exception as error:
        report["operations"][stage] = "pending" if stage == "poll" and isinstance(error, TimeoutError) else "failed"
        report["diagnostics"][stage] = {"error_type": type(error).__name__, "http_status": getattr(error, "status", getattr(getattr(error, "response", None), "status_code", None))}
    finally:
        # Never remove inputs/output while a job is running or creation is uncertain.
        if state.get("terminal_state") in TERMINAL and state.get("bucket_created"):
            try:
                for item in await objects():
                    await storage("DELETE", base + "/o/" + quote(item["name"], safe=""), params={"ifGenerationMatch": item["generation"]})
                await storage("DELETE", base)
                state["cleaned"] = True
                checkpoint()
                report["operations"]["storage-cleanup"] = "passed"
            except Exception as error:
                report["operations"]["storage-cleanup"] = "failed"
                report["diagnostics"]["storage-cleanup"] = {"error_type": type(error).__name__}
        session.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["operations"]), flush=True)
    return int(any(value != "passed" for value in report["operations"].values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.8-flash")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
