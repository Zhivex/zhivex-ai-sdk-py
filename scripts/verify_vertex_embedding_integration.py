"""Exact-wheel Vertex embedding checks with synthetic multimodal inputs."""

from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import wave

import google.auth
from zhivex_ai import create_vertex, FilePart, ImagePart, TextPart
from zhivex_ai.types import RetryOptions
from verify_vertex_chat_integration import red_png
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
    model = provider.embedding_model(args.model)
    report = {
        "schema_version": 1, "provider": "vertex", "model": args.model,
        "mode": "standard-adc-embedding", "location": args.location,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "wheel_sha256": digest, "evidence_status": "integration-only",
        "operations": {}, "diagnostics": {},
    }
    checks = [("text", ["A red circle on a white background."])]
    if args.batch:
        checks.append(("multiple-inputs", ["A red circle.", "A blue square."]))
    if args.image:
        checks.append(("image", [[ImagePart(image=red_png())]]))
    if args.multimodal:
        fixtures = Path(__file__).resolve().parents[1] / "tests/fixtures/vertex"
        report["fixture_sha256"] = {}
        media = [
            ("document", "application/pdf", (fixtures / "document-marker.pdf").read_bytes()),
            ("video", "video/mp4", (fixtures / "color-sequence.mp4").read_bytes()),
        ]
        audio = io.BytesIO()
        with wave.open(audio, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 16000))) for i in range(16000)))
        media.append(("audio", "audio/wav", audio.getvalue()))
        for name, mime, data in media:
            report["fixture_sha256"][name] = hashlib.sha256(data).hexdigest()
            checks.append((name, [[FilePart(data=base64.b64encode(data).decode(), media_type=mime)]]))
        checks.append(("combined-text-image", [[TextPart(text="A red circle."), ImagePart(image=red_png())]]))
    if args.native_config:
        fixtures = Path(__file__).resolve().parents[1] / "tests/fixtures/vertex"
        checks.extend([
            ("native-document-ocr", [[FilePart(data=base64.b64encode((fixtures / "document-marker.pdf").read_bytes()).decode(), media_type="application/pdf")]]),
            ("native-video-without-audio", [[FilePart(data=base64.b64encode((fixtures / "color-sequence.mp4").read_bytes()).decode(), media_type="video/mp4")]]),
        ])
    for name, values in checks:
        try:
            async with asyncio.timeout(60):
                if name.startswith("native-"):
                    part = values[0][0]
                    payload = await provider.native.model_garden().embed_content(
                        model=args.model,
                        body={"content": {"parts": [{"inlineData": {"mimeType": part.media_type, "data": part.data}}]},
                              "embedContentConfig": {"outputDimensionality": 128,
                                  **({"documentOcr": True} if name == "native-document-ocr" else {"audioTrackExtraction": False})}},
                        options=RetryOptions(timeout_ms=45000, max_retries=0),
                    )
                    vectors = [payload["embedding"]["values"]]
                    if payload.get("truncated", False):
                        raise RuntimeError("Unexpected truncation of synthetic fixture")
                else:
                    result = await model.embed(values, {
                        "output_dimensionality": 128, "timeout_ms": 45000,
                        "max_retries": 0,
                    })
                    vectors = result.embeddings
                if len(vectors) != len(values):
                    raise RuntimeError("Embedding cardinality mismatch")
                for vector in vectors:
                    if len(vector) != 128 or not all(math.isfinite(x) for x in vector):
                        raise RuntimeError("Invalid embedding dimensions or values")
                    if not any(vector):
                        raise RuntimeError("Zero embedding vector")
            report["operations"][name] = "passed"
        except Exception as error:
            report["operations"][name] = "failed"
            report["diagnostics"][name] = {
                "error_type": type(error).__name__,
                "http_status": getattr(error, "status", None),
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, report["operations"][name], flush=True)
    return int("failed" in report["operations"].values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--multimodal", action="store_true", help="Check synthetic PDF, video, tone audio and combined text/image inputs.")
    parser.add_argument("--native-config", action="store_true", help="Check native document OCR and disabled video audio extraction.")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
