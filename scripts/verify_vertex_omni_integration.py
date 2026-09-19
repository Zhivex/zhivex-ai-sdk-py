"""One stateless three-second Omni video, with no automatic retry or persisted media."""
from __future__ import annotations

import argparse
import asyncio
import base64
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
    provider = create_vertex(credentials=credentials, project_id=args.project, location="global")
    report = {
        "schema_version": 1, "provider": "vertex", "model": "gemini-omni-1.1-flash-preview",
        "mode": "standard-adc-interactions", "location": "global",
        "recorded_at": datetime.now(timezone.utc).isoformat(), "wheel_sha256": digest,
        "evidence_status": "integration-only", "operations": {}, "diagnostics": {},
    }
    # Reserve output before dispatch: interruption must not silently duplicate generation.
    report["operations"]["video"] = "pending"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        async with asyncio.timeout(240):
            result = await provider.native.interactions().create({
                "model": report["model"], "store": False, "background": False,
                "input": [{"type": "text", "text": "A blue ball rolls slowly across a plain white floor. Static camera. No sound."}],
                "response_format": [{"type": "video", "resolution": "360p", "duration": "3s", "aspect_ratio": "16:9"}],
                "generation_config": {"video_config": {"task": "text_to_video"}},
            }, RetryOptions(timeout_ms=230000, max_retries=0))
            parts = [part for step in result.get("steps", []) for part in step.get("content", [])]
            parts.extend(result.get("outputs", []))
            videos = [part for part in parts if part.get("type") == "video"]
            report["diagnostics"]["status"] = result.get("status")
            report["diagnostics"]["video_count"] = len(videos)
            if not videos:
                raise RuntimeError("No video output")
            inline = [base64.b64decode(part["data"], validate=True) for part in videos if part.get("data")]
            if not inline or not all(len(data) > 1000 and data[4:8] == b"ftyp" for data in inline):
                raise RuntimeError("No verified inline MP4")
            report["operations"]["video"] = "passed"
    except Exception as error:
        report["operations"]["video"] = "failed"
        report["diagnostics"]["error_type"] = type(error).__name__
        report["diagnostics"]["http_status"] = getattr(error, "status", None)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["operations"]), flush=True)
    return int(report["operations"]["video"] != "passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
