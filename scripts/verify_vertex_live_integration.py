"""Bounded installed-wheel Vertex Live audio-turn integration (not release certification)."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import wave

import google.auth
from zhivex_ai import create_vertex, tool
from pydantic import BaseModel
from zhivex_ai.types import AudioFrame, AudioInput, RetryOptions, ToolExecutionResult
from zhivex_ai.live import (
    RealtimeSessionConfig,
    RealtimeAudioOutputEvent,
    RealtimeResponseCompletedEvent,
    RealtimeSessionEndedEvent,
    RealtimeToolCallEvent,
    RealtimeTranscriptEvent,
)
from verify_vertex_integration import verify_wheel


class LookupInput(BaseModel):
    value: str


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
        "mode": "standard-adc-live",
        "location": args.location,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "wheel_sha256": digest,
        "evidence_status": "integration-only",
        "operations": {},
        "diagnostics": {},
        "vad_silence_ms": args.vad_silence_ms,
    }
    operations: dict[str, str] = report["operations"]  # type: ignore[assignment]
    diagnostics: dict[str, object] = report["diagnostics"]  # type: ignore[assignment]
    session = None
    current = "setup"
    audio_bytes = 0
    tool_seen = False
    completion_reason = None
    transcript = ""
    input_transcript = ""
    def markers_match(text: str) -> bool:
        normalized = "".join(character for character in text.lower() if character.isalpha())
        return all(word in normalized for word in ("turquoise", "notebook"))

    try:
        async with asyncio.timeout(210 if args.audio_input else 75):
            speech = None
            if args.audio_input:
                current = "tts-fixture"
                speech = await create_vertex(credentials=credentials, project_id=args.project, location="global").speech_model("gemini-3.1-flash-tts-preview").generate_speech(
                    input="Say exactly in one short continuous sentence without pauses: Repeat turquoise notebook.",
                    voice="Kore", options=RetryOptions(timeout_ms=60000, max_retries=0),
                )
                if not speech.audio or "rate=24000" not in speech.media_type.lower() or not speech.media_type.lower().startswith(("audio/l16", "audio/pcm")):
                    raise RuntimeError("Expected 24 kHz PCM fixture")
                report["audio_fixture"] = {
                    "sha256": hashlib.sha256(speech.audio).hexdigest(),
                    "bytes": len(speech.audio), "sample_rate_hz": 24000,
                    "speech_model": "gemini-3.1-flash-tts-preview",
                    "transcription_model": "gemini-3.5-transcribe-preview",
                }
                operations[current] = "passed"
                current = "tts-fixture-transcription"
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(speech.audio)
                reference = await create_vertex(credentials=credentials, project_id=args.project, location="global").transcription_model("gemini-3.5-transcribe-preview").transcribe(
                    audio=AudioInput(data=buffer.getvalue(), media_type="audio/wav"),
                    options=RetryOptions(timeout_ms=60000, max_retries=0),
                )
                if not markers_match(reference.text):
                    raise RuntimeError("Synthetic audio fixture did not contain the expected markers")
                operations[current] = "passed"
                current = "setup"
            session = await provider.realtime_model(args.model).connect(
                RealtimeSessionConfig(
                    instructions="Keep your spoken responses very short.",
                    input_audio_media_type="audio/pcm;rate=24000" if args.audio_input else None,
                    input_sample_rate_hz=24000 if args.audio_input else None,
                    output_audio_media_type="audio/pcm",
                    output_sample_rate_hz=24000,
                    provider_options={"outputAudioTranscription": {}, **({"inputAudioTranscription": {},
                        **({"realtimeInputConfig": {"automaticActivityDetection": {"silenceDurationMs": args.vad_silence_ms}}} if args.vad_silence_ms is not None else {})
                    } if args.audio_input else {})},
                    tools={
                        "lookup": tool(
                            name="lookup", schema=LookupInput, execute=lambda x: x
                        )
                    }
                    if args.tools
                    else None,
                )
            )
            operations["setup"] = "passed"
            current = "audio-to-audio-turn" if args.audio_input else "tool-to-audio-turn" if args.tools else "text-to-audio-turn"
            if speech is not None:
                for offset in range(0, len(speech.audio), 4800):
                    end = offset + 4800
                    await session.send_audio(AudioFrame(data=speech.audio[offset:end], media_type="audio/pcm;rate=24000", is_final=end >= len(speech.audio)))
                    await asyncio.sleep(0.1)
            else:
                await session.send_text(
                    "Call lookup with value VERTEX_LIVE_OK. Do not answer before calling it. Then speak the result it returns."
                    if args.tools
                    else "Say hello."
                )
            complete = False
            async for event in session.event_stream():
                if isinstance(event, RealtimeToolCallEvent):
                    call = event.tool_call
                    if (
                        not args.tools
                        or call.name != "lookup"
                        or call.input != {"value": "VERTEX_LIVE_OK"}
                    ):
                        raise RuntimeError("Unexpected live tool call")
                    tool_seen = True
                    await session.send_tool_result(
                        ToolExecutionResult(
                            tool_call_id=call.id,
                            tool_name=call.name,
                            output={"result": "The secret word is mango."},
                        )
                    )
                if isinstance(event, RealtimeAudioOutputEvent):
                    audio_bytes += len(event.audio)
                elif isinstance(event, RealtimeTranscriptEvent) and event.role == "user":
                    input_transcript += event.text
                elif (
                    isinstance(event, RealtimeTranscriptEvent)
                    and event.role == "assistant"
                ):
                    transcript += event.text
                elif isinstance(event, RealtimeResponseCompletedEvent):
                    completion_reason = event.reason
                    # A tool-call turn may finish before the function result
                    # produces the assistant's subsequent audio turn.
                    if (
                        event.reason == "turn-complete"
                        and audio_bytes
                        and (
                            not args.tools
                            or (tool_seen and "mango" in transcript.lower())
                        )
                    ):
                        if args.audio_input and not (markers_match(transcript) and markers_match(input_transcript)):
                            raise RuntimeError("Completed audio turn did not preserve the fixture markers")
                        complete = True
                        break
                elif isinstance(event, RealtimeSessionEndedEvent):
                    raise RuntimeError("Session ended before a complete audio turn")
            if not complete or not audio_bytes or (args.tools and not tool_seen):
                raise RuntimeError("No complete audio turn")
            operations[current] = "passed"
    except Exception as error:
        operations[current] = "failed"
        diagnostics[current] = {
            "error_type": type(error).__name__,
            "http_status": getattr(error, "status", None),
            "websocket_close_code": getattr(getattr(error, "rcvd", None), "code", None),
            "audio_received": bool(audio_bytes),
            "tool_seen": tool_seen,
            "completion_reason": completion_reason,
            "input_markers_match": markers_match(input_transcript) if args.audio_input else None,
            "output_markers_match": markers_match(transcript) if args.audio_input else None,
            "input_transcript_characters": len(input_transcript) if args.audio_input else None,
            "input_transcript_word_count": len(input_transcript.split()) if args.audio_input else None,
        }
    finally:
        if session is not None:
            try:
                async with asyncio.timeout(10):
                    await session.aclose()
                operations["cleanup"] = "passed"
            except Exception as error:
                operations["cleanup"] = "failed"
                diagnostics["cleanup"] = {"error_type": type(error).__name__}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(operations), flush=True)
    return int(any(value != "passed" for value in operations.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model", default="gemini-live-2.5-flash-native-audio")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tools", action="store_true")
    mode.add_argument("--audio-input", action="store_true")
    parser.add_argument("--vad-silence-ms", type=int, choices=range(0, 2001), metavar="0..2000")
    args = parser.parse_args()
    if args.vad_silence_ms is not None and not args.audio_input:
        parser.error("--vad-silence-ms requires --audio-input")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
