"""Single-job, resumable exact-wheel Vertex Veo integration check."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
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


async def run(args: argparse.Namespace) -> int:
    digest = verify_wheel(args.wheel)
    if args.output.exists():
        raise SystemExit("Choose a fresh evidence output path.")
    identity = {"project": args.project, "location": args.location, "model": args.model}
    references = getattr(args, "reference", None) or []
    video = getattr(args, "video", None)
    if video and (args.image or args.last_frame or references):
        raise SystemExit("Video extension cannot be combined with images or references.")
    video_data = video.read_bytes() if video else None
    if video_data is not None:
        if len(video_data) < 100 or video_data[4:8] != b"ftyp" or len(video_data) > 20 * 1024 * 1024:
            raise SystemExit("Use an MP4 input of at most 20 MB.")
        identity["video_sha256"] = hashlib.sha256(video_data).hexdigest()
    if references and (args.image or args.last_frame):
        raise SystemExit("Reference images cannot be combined with first/last frames.")
    if len(references) > 3:
        raise SystemExit("Use at most three asset reference images.")
    reference_data = [path.read_bytes() for path in references]
    if reference_data:
        if any(not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) > 20 * 1024 * 1024 for data in reference_data):
            raise SystemExit("Use PNG reference images of at most 20 MB each.")
        identity["reference_sha256"] = [hashlib.sha256(data).hexdigest() for data in reference_data]
    if args.last_frame and not args.image:
        raise SystemExit("--last-frame requires --image.")
    image_data = args.image.read_bytes() if args.image else None
    last_frame_data = args.last_frame.read_bytes() if args.last_frame else None
    if last_frame_data is not None:
        if not last_frame_data.startswith(b"\x89PNG\r\n\x1a\n") or len(last_frame_data) > 20 * 1024 * 1024:
            raise SystemExit("Use a PNG last frame of at most 20 MB.")
        identity["last_frame_sha256"] = hashlib.sha256(last_frame_data).hexdigest()
    if image_data is not None:
        if not image_data.startswith(b"\x89PNG\r\n\x1a\n") or len(image_data) > 20 * 1024 * 1024:
            raise SystemExit("Use a PNG input of at most 20 MB.")
        identity["image_sha256"] = hashlib.sha256(image_data).hexdigest()
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state.get("identity") != identity or state.get("wheel_sha256") != digest:
            raise SystemExit("Checkpoint belongs to another target or wheel.")
        if not state.get("operation_name"):
            raise SystemExit("Creation outcome unknown; reconcile the existing request before creating another job.")
    else:
        state = {"identity": identity, "wheel_sha256": digest}
        # Reserve the checkpoint before submitting: never silently duplicate a job
        # if the server accepted it but the creation response was interrupted.
        with args.state.open("x") as file:
            file.write(json.dumps(state))
        args.state.chmod(0o600)
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
        quota_project_id=args.project,
    )
    client = create_vertex(credentials=credentials, project_id=args.project, location=args.location).videos()
    report = {
        "schema_version": 1, "provider": "vertex", "model": args.model,
        "mode": "standard-adc-video", "location": args.location,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "wheel_sha256": digest, "evidence_status": "integration-only",
        "operations": {}, "diagnostics": {},
        "video_sha256": identity.get("video_sha256"),
        "input_mode": "video-extension" if video_data is not None else "asset-reference" if reference_data else "first-last-frame" if last_frame_data is not None else "image-to-video" if image_data is not None else "text-to-video",
        "image_sha256": identity.get("image_sha256"),
        "last_frame_sha256": identity.get("last_frame_sha256"),
        "reference_sha256": identity.get("reference_sha256"),
    }
    stage = "create"
    try:
        if not state.get("operation_name"):
            operation = await client.generate(
                model=args.model,
                prompt="Continue the gentle camera motion and preserve the red circle on the white background. No people or text." if video_data is not None else "Animate the red circle from the reference image with a slow gentle camera zoom on a plain white background. No people or text." if reference_data else "Animate this image with a slow gentle camera zoom. Preserve its colors. No people or text." if image_data is not None else "A red wooden cube slowly rotating on a plain white table. Static camera. No people or text.",
                config={**({} if video_data is not None else {"durationSeconds": 8 if reference_data else 4}), "sampleCount": 1, "resolution": "720p", "generateAudio": False},
                extra_body={"instance": {"video": {"bytesBase64Encoded": base64.b64encode(video_data).decode(), "mimeType": "video/mp4"}}} if video_data is not None else {"instance": {
                    "image": {"bytesBase64Encoded": base64.b64encode(image_data).decode(), "mimeType": "image/png"},
                    **({"lastFrame": {"bytesBase64Encoded": base64.b64encode(last_frame_data).decode(), "mimeType": "image/png"}} if last_frame_data is not None else {}),
                }} if image_data is not None else {"instance": {"referenceImages": [
                    {"image": {"bytesBase64Encoded": base64.b64encode(data).decode(), "mimeType": "image/png"}, "referenceType": "asset"}
                    for data in reference_data
                ]}} if reference_data else None,
                options=RetryOptions(timeout_ms=60000, max_retries=0),
            )
            if not operation.name:
                raise RuntimeError("Missing operation name")
            state["operation_name"] = operation.name
            args.state.write_text(json.dumps(state))
        report["operations"]["create"] = "passed"
        print("create passed; checkpoint saved", flush=True)
        stage = "poll"
        operation = await client.wait_operation(state["operation_name"], poll_interval_ms=10000, timeout_ms=240000)
        if operation.error:
            report["diagnostics"]["operation_error_code"] = operation.error.get("code")
            raise RuntimeError("Operation terminated with an error")
        report["operations"]["poll"] = "passed"
        stage = "video-payload"
        media = operation.raw_response.get("generated_media", [])
        if len(media) != 1:
            raise RuntimeError("Expected one generated video")
        item = media[0]
        if item.b64_data:
            data = base64.b64decode(item.b64_data, validate=True)
        elif item.url:
            data = await client.download(item.url, RetryOptions(timeout_ms=60000, max_retries=0))
            report["operations"]["download"] = "passed"
        else:
            raise RuntimeError("Missing video data or URI")
        if item.media_type != "video/mp4" or len(data) < 100 or data[4:8] != b"ftyp":
            raise RuntimeError("Invalid MP4 payload")
        report["operations"]["video-payload"] = "passed"
        report["diagnostics"]["output_bytes"] = len(data)
        report["diagnostics"]["output_sha256"] = hashlib.sha256(data).hexdigest()
        state["terminal"] = True
        args.state.write_text(json.dumps(state))
    except Exception as error:
        report["operations"][stage] = "pending" if stage == "poll" and isinstance(error, TimeoutError) else "failed"
        report["diagnostics"][stage] = {"error_type": type(error).__name__, "http_status": getattr(error, "status", None)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["operations"]), flush=True)
    return int(any(value != "passed" for value in report["operations"].values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model", default="veo-3.1-fast-generate-001")
    parser.add_argument("--video", type=Path, help="MP4 extension input; cannot combine with images; digest bound to checkpoint")
    parser.add_argument("--image", type=Path, help="Optional PNG first-frame fixture; its digest is bound to the checkpoint")
    parser.add_argument("--last-frame", type=Path, help="Optional PNG last frame; requires --image and binds its digest to the checkpoint")
    parser.add_argument("--reference", type=Path, action="append", help="PNG asset reference, up to three; cannot combine with first/last frames; uses eight seconds")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
