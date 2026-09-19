"""Synthetic document and TTS-to-audio-input checks on an exact Vertex wheel."""

from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import wave

import google.auth
from zhivex_ai import create_vertex, generate_text, FilePart, ModelMessage, TextPart, AudioInput
from zhivex_ai.types import RetryOptions
from verify_vertex_integration import verify_wheel


async def run(args: argparse.Namespace) -> int:
    digest = verify_wheel(args.wheel)
    if args.output.exists():
        raise SystemExit("Choose a fresh evidence path.")
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=args.project)
    provider = create_vertex(credentials=credentials, project_id=args.project, location="global")
    report = {
        "schema_version": 1, "provider": "vertex", "model": args.model,
        "speech_model": args.speech_model, "mode": "standard-adc-multimodal",
        "location": "global", "recorded_at": datetime.now(timezone.utc).isoformat(),
        "wheel_sha256": digest, "evidence_status": "integration-only",
        "operations": {}, "diagnostics": {},
    }
    audio = None

    async def document():
        pdf = Path(__file__).resolve().parents[1] / "tests/fixtures/vertex/document-marker.pdf"
        result = await generate_text(
            model=provider(args.model),
            messages=[ModelMessage(role="user", parts=[
                TextPart(text="Read the attached document. Return its verification code and preferred color."),
                FilePart(data=base64.b64encode(pdf.read_bytes()).decode(), media_type="application/pdf"),
            ])], max_tokens=512, timeout_ms=60000, max_retries=0,
        )
        if "ORCHID-7284" not in result.text or "turquoise" not in result.text.lower():
            raise RuntimeError("Document marker mismatch")

    async def speech():
        nonlocal audio
        result = await provider.speech_model(args.speech_model).generate_speech(
            input="Say exactly: The secret words are turquoise notebook.", voice="Kore",
            options=RetryOptions(timeout_ms=60000, max_retries=0),
        )
        report["diagnostics"]["speech_media_type"] = result.media_type
        if result.media_type.lower().startswith(("audio/l16", "audio/pcm")):
            match = re.search(r"rate=(\d+)", result.media_type)
            if not match:
                raise RuntimeError("PCM output omitted sample rate")
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as file:
                file.setnchannels(1)
                file.setsampwidth(2)
                file.setframerate(int(match.group(1)))
                file.writeframes(result.audio)
            audio = AudioInput(data=buffer.getvalue(), media_type="audio/wav")
        else:
            audio = AudioInput(data=result.audio, media_type=result.media_type)
        if not result.audio:
            raise RuntimeError("Empty speech")

    async def transcribe():
        if audio is None:
            raise RuntimeError("Speech prerequisite failed")
        result = await provider.transcription_model(args.model).transcribe(
            audio=audio, prompt=None if args.transcription_only else "Transcribe the spoken sentence verbatim.",
            options=RetryOptions(timeout_ms=60000, max_retries=0),
        )
        if not all(word in result.text.lower() for word in ("turquoise", "notebook")):
            raise RuntimeError("Audio-only marker mismatch")

    checks = [("tts-fixture", speech), ("audio-transcription", transcribe)]
    if not args.transcription_only:
        checks.insert(0, ("pdf-input", document))
    for name, call in checks:
        try:
            async with asyncio.timeout(90):
                await call()
            report["operations"][name] = "passed"
        except Exception as error:
            report["operations"][name] = "failed"
            report["diagnostics"][name] = {"error_type": type(error).__name__, "http_status": getattr(error, "status", None)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, report["operations"][name], flush=True)
    return int("failed" in report["operations"].values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--transcription-only", action="store_true")
    parser.add_argument("--speech-model", default="gemini-3.1-flash-tts-preview")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
