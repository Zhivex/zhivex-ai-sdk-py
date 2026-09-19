"""Exact-wheel synthetic English-to-Spanish Live speech translation integration check."""
from __future__ import annotations

import argparse
from array import array
import sys
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path

import google.auth
from zhivex_ai import create_vertex, AudioFrame
from zhivex_ai.live import RealtimeSessionConfig, RealtimeTranscriptEvent, RealtimeAudioOutputEvent
from zhivex_ai.types import RetryOptions
from verify_vertex_integration import verify_wheel


async def run(args):
    digest = verify_wheel(args.wheel)
    if args.output.exists():
        raise SystemExit("Choose a fresh evidence path.")
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=args.project)
    provider = create_vertex(credentials=credentials, project_id=args.project, location="global")
    report = {
        "schema_version": 1, "provider": "vertex", "model": "gemini-3.5-live-translate-preview",
        "mode": "standard-adc-live", "location": "global",
        "recorded_at": datetime.now(timezone.utc).isoformat(), "wheel_sha256": digest,
        "evidence_status": "integration-only", "operations": {}, "diagnostics": {},
    }
    session = None
    stage = "tts-fixture"
    try:
        async with asyncio.timeout(100):
            speech = await provider.speech_model("gemini-3.1-flash-tts-preview").generate_speech(
                input="Say exactly: The blue cat sleeps on the table.", voice="Kore",
                options=RetryOptions(timeout_ms=60000, max_retries=0),
            )
            if not speech.audio or "rate=24000" not in speech.media_type.lower() or not speech.media_type.lower().startswith(("audio/l16", "audio/pcm")):
                raise RuntimeError("Expected 24 kHz PCM fixture")
            report["operations"][stage] = "passed"
            samples = array("h", speech.audio)
            if sys.byteorder != "little":
                samples.byteswap()
            # Linear resampling is sufficient for this short synthetic speech fixture.
            downsampled = array("h", (int((samples[i] + samples[min(i + 1, len(samples) - 1)]) / 2) if n % 2 else samples[i] for n in range(len(samples) * 2 // 3) for i in [n * 3 // 2]))
            if sys.byteorder != "little":
                downsampled.byteswap()
            pcm = downsampled.tobytes() + bytes(32000)  # one second of trailing silence for VAD
            stage = "setup"
            session = await provider.realtime_model(report["model"]).connect(RealtimeSessionConfig(
                input_audio_media_type="audio/pcm;rate=16000", input_sample_rate_hz=16000,
                output_audio_media_type="audio/pcm", translation_target_language_code="es",
                translation_echo_target_language=False,
                provider_options={"inputAudioTranscription": {}, "outputAudioTranscription": {}},
            ))
            report["operations"][stage] = "passed"
            stage = "audio-translation"

            async def send():
                for offset in range(0, len(pcm), 3200):
                    end = offset + 3200
                    await session.send_audio(AudioFrame(
                        data=pcm[offset:end], media_type="audio/pcm;rate=16000",
                        is_final=end >= len(pcm),
                    ))
                    await asyncio.sleep(0.1)

            sender = asyncio.create_task(send())
            final = ""
            audio_received = 0
            try:
                async for event in session.event_stream():
                    counts = report["diagnostics"].setdefault("event_counts", {})
                    counts[event.type] = counts.get(event.type, 0) + 1
                    if isinstance(event, RealtimeTranscriptEvent) and event.role == "user":
                        report["diagnostics"]["source_transcript_received"] = True
                    if isinstance(event, RealtimeAudioOutputEvent):
                        audio_received += len(event.audio)
                        report["diagnostics"]["translated_audio_received"] = True
                    if isinstance(event, RealtimeTranscriptEvent) and event.role == "assistant":
                        final += event.text
                        report["diagnostics"]["spanish_markers"] = [word for word in ("gato", "azul", "mesa") if word in final.lower()]
                    if audio_received and all(word in final.lower() for word in ("gato", "azul", "mesa")):
                        break
                await sender
            finally:
                if not sender.done():
                    sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            report["diagnostics"]["translated_audio_received"] = bool(audio_received)
            if not audio_received or not all(word in final.lower() for word in ("gato", "azul", "mesa")):
                raise RuntimeError("Final transcript marker mismatch")
            report["operations"][stage] = "passed"
    except Exception as error:
        report["operations"][stage] = "failed"
        report["diagnostics"][stage] = {"error_type": type(error).__name__, "http_status": getattr(error, "status", None), "websocket_close_code": getattr(getattr(error, "rcvd", None), "code", None)}
    finally:
        if session is not None:
            try:
                async with asyncio.timeout(10):
                    await session.aclose()
                report["operations"]["cleanup"] = "passed"
            except Exception as error:
                report["operations"]["cleanup"] = "failed"
                report["diagnostics"]["cleanup"] = {"error_type": type(error).__name__}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["operations"]), flush=True)
    return int("failed" in report["operations"].values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
