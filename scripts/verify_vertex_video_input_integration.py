"""Verify temporal video understanding with a locally validated synthetic clip."""
from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import google.auth
from pydantic import BaseModel
from zhivex_ai import create_vertex, generate_object, FilePart, ModelMessage, TextPart
from verify_vertex_integration import verify_wheel


class Sequence(BaseModel):
    colors: list[str]


async def run(args):
    digest = verify_wheel(args.wheel)
    if args.output.exists():
        raise SystemExit("Choose a fresh evidence path.")
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/vertex/color-sequence.mp4"
    data = fixture.read_bytes()
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=args.project)
    provider = create_vertex(credentials=credentials, project_id=args.project, location="global")
    report = {
        "schema_version": 1, "provider": "vertex", "model": args.model,
        "mode": "standard-adc-video-input", "location": "global",
        "recorded_at": datetime.now(timezone.utc).isoformat(), "wheel_sha256": digest,
        "fixture_sha256": hashlib.sha256(data).hexdigest(),
        "evidence_status": "integration-only", "operations": {}, "diagnostics": {},
    }
    try:
        async with asyncio.timeout(90):
            result = await generate_object(
                model=provider(report["model"]),
                messages=[ModelMessage(role="user", parts=[
                    TextPart(text="List the distinct full-screen colors in their temporal order in this video. Use lowercase English color names, one entry per continuous color segment."),
                    FilePart(data=base64.b64encode(data).decode(), media_type="video/mp4"),
                ])], schema=Sequence, max_tokens=512, timeout_ms=80000, max_retries=0,
            )
            if result.object.colors != ["red", "blue", "green"]:
                raise RuntimeError("Temporal sequence mismatch")
            report["operations"]["video-temporal-sequence"] = "passed"
    except Exception as error:
        report["operations"]["video-temporal-sequence"] = "failed"
        report["diagnostics"] = {"error_type": type(error).__name__, "http_status": getattr(error, "status", None)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["operations"]), flush=True)
    return int("failed" in report["operations"].values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--model", default="gemini-3.8-flash")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
