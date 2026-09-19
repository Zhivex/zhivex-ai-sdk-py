"""Bounded exact-wheel Gemma/Vertex Chat Completions integration checks."""

from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import struct
import zlib

import google.auth
from pydantic import BaseModel
from zhivex_ai import (
    create_vertex,
    generate_text,
    generate_object,
    stream_text,
    ModelMessage,
    TextPart,
    ImagePart,
)
from verify_vertex_integration import verify_wheel


class Output(BaseModel):
    value: str


def red_png() -> str:
    """Deterministic synthetic 64x64 RGB fixture; no external media or user data."""

    def chunk(kind: bytes, value: bytes) -> bytes:
        return (
            struct.pack(">I", len(value))
            + kind
            + value
            + struct.pack(">I", zlib.crc32(kind + value))
        )

    raw = (b"\x00" + b"\xff\x00\x00" * 64) * 64
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


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
    if args.responses:
        model = provider.native.model_garden().responses_model(args.model)
    else:
        model = provider.native.model_garden().language_model(args.model) if args.garden else provider(args.model)
    os.environ["ZHIVEX_SMOKE_USE_INSTALLED"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.run_live_smoke import _run_agent_tool_smoke

    native_options = {"chat_template_kwargs": {"enable_thinking": False}} if args.disable_thinking else None

    async def generation() -> None:
        result = await generate_text(
            model=model,
            prompt="Reply with exactly GEMMA_OK.",
            max_tokens=args.max_tokens,
            provider_options=native_options,
            timeout_ms=60000,
            max_retries=0,
        )
        diagnostics["generation-result"] = {"finish_reason": result.finish_reason, "text_chars": len(result.text), "output_tokens": result.usage.output_tokens if result.usage else None}
        if "GEMMA_OK" not in result.text:
            raise RuntimeError("Generation marker mismatch")

    async def streaming() -> None:
        result = await stream_text(
            model=model,
            prompt="Reply with exactly GEMMA_OK.",
            max_tokens=args.max_tokens,
            provider_options=native_options,
            timeout_ms=60000,
            max_retries=0,
        ).collect()
        diagnostics["streaming-result"] = {"finish_reason": result.finish_reason, "text_chars": len(result.text), "output_tokens": result.usage.output_tokens if result.usage else None}
        if "GEMMA_OK" not in result.text:
            raise RuntimeError("Stream marker mismatch")

    async def structured() -> None:
        result = await generate_object(
            model=model,
            prompt="Return JSON with value exactly GEMMA_OK.",
            schema=Output,
            max_tokens=args.max_tokens,
            timeout_ms=60000,
            max_retries=0,
        )
        if result.object.value != "GEMMA_OK":
            raise RuntimeError("JSON marker mismatch")

    report = {
        "schema_version": 1,
        "provider": "vertex",
        "model": args.model,
        "mode": "standard-adc-language",
        "adapter": type(model).__name__,
        "location": args.location,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "wheel_sha256": digest,
        "evidence_status": "integration-only",
        "operations": {},
        "diagnostics": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    operations: dict[str, str] = report["operations"]  # type: ignore[assignment]
    diagnostics: dict[str, object] = report["diagnostics"]  # type: ignore[assignment]

    async def vision() -> None:
        result = await generate_text(
            model=model,
            messages=[
                ModelMessage(
                    role="user",
                    parts=[
                        TextPart(
                            text="Name the dominant color in this image. Reply with one English word."
                        ),
                        ImagePart(image=red_png()),
                    ],
                )
            ],
            max_tokens=128,
            timeout_ms=60000,
            max_retries=0,
        )
        if "red" not in result.text.lower():
            raise RuntimeError("Vision fixture mismatch")

    checks = [
        ("generation", generation),
        ("streaming", streaming),
    ]
    if not args.basic:
        checks.extend([
            ("structured-output", structured),
            ("agent-tool", lambda: _run_agent_tool_smoke(provider="vertex", model=model)),
        ])
    if args.vision_only:
        checks = []
    if args.vision or args.vision_only:
        checks.append(("vision", vision))
    if args.operation:
        selected = set(args.operation)
        if not selected <= {name for name, _ in checks}:
            raise SystemExit("Selected operation is disabled by the current basic/vision flags.")
        checks = [(name, call) for name, call in checks if name in selected]
    report["selected_operations"] = [name for name, _ in checks]
    report["interval_seconds"] = args.interval_seconds
    report["disable_thinking"] = args.disable_thinking
    report["max_tokens"] = args.max_tokens
    for index, (name, call) in enumerate(checks):
        if index and args.interval_seconds:
            await asyncio.sleep(args.interval_seconds)
        try:
            async with asyncio.timeout(90):
                await call()
            operations[name] = "passed"
        except Exception as error:
            operations[name] = "failed"
            diagnostics[name] = {
                "error_type": type(error).__name__,
                "http_status": getattr(error, "status", None),
            }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}: {operations[name]}", flush=True)
    return int(any(value != "passed" for value in operations.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="global")
    parser.add_argument("--model", default="google/gemma-4-26b-a4b-it-maas")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--vision-only", action="store_true")
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--garden", action="store_true", help="Use the explicit Model Garden Chat Completions adapter")
    route.add_argument("--responses", action="store_true", help="Use normalized Model Garden Responses; use --basic unless capabilities have been verified")
    parser.add_argument("--disable-thinking", action="store_true", help="GLM 5.2 native basic checks only; sends documented enable_thinking=false")
    parser.add_argument("--basic", action="store_true", help="Verify text and streaming only; do not infer other capabilities")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--operation", action="append", choices=("generation", "streaming", "structured-output", "agent-tool", "vision"))
    parser.add_argument("--interval-seconds", type=int, choices=range(61), default=0, help="Optional delay between selected operations, up to 60 seconds")
    args = parser.parse_args()
    if args.disable_thinking and (not args.garden or not args.basic or args.model != "zai-org/glm-5.2-maas"):
        parser.error("--disable-thinking requires --garden --basic and zai-org/glm-5.2-maas")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
