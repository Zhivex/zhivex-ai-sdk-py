"""Bounded synthetic image/speech checks against an exact installed Vertex wheel."""

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


async def run(args: argparse.Namespace) -> int:
    digest = verify_wheel(args.wheel)
    if args.output.exists():
        raise SystemExit("Choose a fresh evidence path.")
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
        quota_project_id=args.project,
    )
    provider = create_vertex(
        credentials=credentials, project_id=args.project, location=args.location
    )
    report = {
        "schema_version": 1,
        "provider": "vertex",
        "model": args.model,
        "mode": "standard-adc-media",
        "location": args.location,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "wheel_sha256": digest,
        "evidence_status": "integration-only",
        "operations": {},
        "diagnostics": {},
        "include_thoughts": args.include_thoughts,
    }
    name = "image-edit" if args.kind == "image-edit" else args.kind + "-generation"
    try:
        async with asyncio.timeout(240 if args.kind == "music" else 120):
            if args.kind in {"image", "image-edit"}:
                extra = {"generationConfig": {"thinkingConfig": {"includeThoughts": True, "thinkingLevel": "high"}}} if args.include_thoughts else {}
                if args.kind == "image-edit":
                    from verify_vertex_chat_integration import red_png

                    result = await provider.images().edit(
                        model=args.model,
                        prompt="Add a blue circle in the center of this red image. No text.",
                        image=base64.b64decode(red_png().split(",", 1)[1]),
                        image_media_type="image/png",
                        extra_body=extra,
                    )
                else:
                    result = await provider.images().generate(
                        model=args.model,
                        prompt="A simple red circle on a plain white background. No text.",
                        extra_body={"parameters": {"sampleCount": 1}, **extra},
                    )
                raw = result.raw_response or {}
                thought_images = [part for candidate in raw.get("candidates", [])
                                  for part in (candidate.get("content") or {}).get("parts", [])
                                  if part.get("thought") is True and (part.get("inlineData") or part.get("inline_data"))]
                report["diagnostics"]["thought_image_count"] = len(thought_images)
                if any(item.metadata.get("thought") is True for item in result.images):
                    raise RuntimeError("Intermediate thought image exposed as final output")
                report["diagnostics"]["image_count"] = len(result.images)
                if len(result.images) != 1:
                    raise RuntimeError("Expected exactly one image")
                item = result.images[0]
                data = base64.b64decode(item.b64_json or "", validate=True)
                png = data.startswith(b"\x89PNG\r\n\x1a\n")
                jpeg = data.startswith(b"\xff\xd8\xff")
                webp = data.startswith(b"RIFF") and data[8:12] == b"WEBP"
                report["diagnostics"]["image_payload"] = {
                    "bytes": len(data), "png": png, "jpeg": jpeg, "webp": webp,
                    "media_type": item.media_type,
                }
                if len(data) < 100 or not (png or jpeg or webp):
                    raise RuntimeError("Invalid image payload")
                if not (item.media_type or "").startswith("image/"):
                    raise RuntimeError("Invalid image media type")
            elif args.kind == "music":
                result = await provider.media().generate_music(
                    model=args.model,
                    prompt="A calm acoustic folk song with a gentle guitar melody and soft strings.",
                    provider_options={"parameters": {"sample_count": 1}} if args.model == "lyria-002" else None,
                    options=RetryOptions(timeout_ms=210000, max_retries=0),
                )
                if len(result.media) != 1:
                    def shape(value):
                        if isinstance(value, dict):
                            return {key: shape(item) for key, item in value.items()}
                        if isinstance(value, list):
                            return [shape(item) for item in value[:2]]
                        return type(value).__name__

                    report["diagnostics"]["response_shape"] = shape(result.raw_response)
                    raise RuntimeError("Expected exactly one audio result")
                item = result.media[0]
                data = base64.b64decode(item.b64_data or "", validate=True)
                wav = data.startswith(b"RIFF") and data[8:12] == b"WAVE"
                mp3 = data.startswith(b"ID3") or (len(data) > 2 and data[0] == 255 and data[1] & 224 == 224)
                if len(data) < 100 or not (wav or mp3) or not item.media_type.startswith("audio/"):
                    raise RuntimeError("Invalid or non-inline audio payload")
            else:
                speech = await provider.speech_model(args.model).generate_speech(
                    input="Say cheerfully: Hello, world.",
                    voice="Kore",
                    options=RetryOptions(timeout_ms=90000, max_retries=0),
                )
                if len(speech.audio) < 100 or not speech.media_type.startswith("audio/"):
                    raise RuntimeError("Invalid audio payload")
                if not any(speech.audio):
                    raise RuntimeError("Silent audio payload")
        report["operations"][name] = "passed"
    except Exception as error:
        report["operations"][name] = "failed"
        report["diagnostics"][name] = {
            "error_type": type(error).__name__,
            "http_status": getattr(error, "status", None),
        }
        try:
            upstream = json.loads(getattr(error, "response_body", None) or "{}").get("error", {})
            message = str(upstream.get("message", "")).lower()
            report["diagnostics"][name]["error_categories"] = [
                token for token in ("safety", "recitation", "sample_count", "seed", "quota", "retired", "not found", "copyright", "blocked")
                if token in message
            ]
        except (ValueError, TypeError, AttributeError):
            pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["operations"]), flush=True)
    return int("failed" in report["operations"].values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="global")
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=["image", "image-edit", "speech", "music"], required=True)
    parser.add_argument("--include-thoughts", action="store_true")
    args = parser.parse_args()
    if args.include_thoughts and (args.kind not in {"image", "image-edit"} or not args.model.startswith("gemini-3")):
        parser.error("--include-thoughts requires a Gemini 3 image model and image or image-edit kind")
    raise SystemExit(asyncio.run(run(args)))
