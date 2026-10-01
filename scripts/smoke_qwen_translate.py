"""Bounded Qwen LiveTranslate integration evidence, optionally from an exact installed wheel.

Uses synthetic speech fixtures. Never persists audio, transcripts, keys, workspace IDs,
or raw provider errors. A successful smoke is not protected release certification.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
if os.getenv("ZHIVEX_SMOKE_USE_INSTALLED") != "1":
    sys.path.insert(0, str(ROOT / "src"))

from zhivex_ai import AudioFrame, create_qwen  # noqa: E402
from zhivex_ai.live import RealtimeConnectOptions, RealtimeSessionConfig  # noqa: E402

MODEL = "qwen3.8-livetranslate-flash-realtime"


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError("Fixture must be mono PCM16 WAV at 16000 Hz.")
        pcm = source.readframes(source.getnframes())
    if not pcm or len(pcm) > 32_000 * 30:
        raise ValueError("Fixture must contain between zero and thirty seconds of speech.")
    return pcm


async def scenario(provider, pcm: bytes, *, audio: bool, clone: bool = False, image: bytes | None = None, min_segments: int = 1) -> dict:
    result: dict = {"status": "failed", "checks": {}, "event_counts": {}}
    session = None
    try:
        async with asyncio.timeout(90):
            options: dict = {"output_modalities": ["text", "audio"] if audio else ["text"],
                             "translation": {"corpus": {"phrases": {"orbit seven": "órbita siete"}}}}
            if clone:
                options.update(enable_voice_clone=True, voice_clone_options={"frequency": "always"})
            session = await provider.native.realtime_model(MODEL).connect(
                RealtimeSessionConfig(translation_target_language_code="es", provider_options=options),
                RealtimeConnectOptions(timeout_ms=30_000),
            )
            result["checks"]["setup_ack"] = True
            translated = []
            source_final = []
            source_items: set[str] = set()
            audio_bytes = 0
            finished = False
            errors = False
            correlated = False

            async def send():
                for offset in range(0, len(pcm), 3200):
                    await session.send_audio(AudioFrame(pcm[offset:offset + 3200], "audio/pcm", sample_rate_hz=16000))
                    if image is not None and offset == 0:
                        await session.send_image(image)
                    await asyncio.sleep(0.1)
                await session.finish()

            async def receive_native():
                async for event in session.native_event_stream():
                    kind = event["type"]
                    counts = result["event_counts"]
                    counts[kind] = counts.get(kind, 0) + 1
                    # Error codes alone are safe diagnostics; freeform messages are excluded.
                    if kind == "error":
                        code = event.get("error", {}).get("code")
                        if isinstance(code, str) and len(code) < 100 and all(c.isalnum() or c in "_-" for c in code):
                            result["provider_error_code"] = code

            async def receive():
                nonlocal audio_bytes, finished, errors, correlated
                async for event in session.event_stream():
                    if event.type == "realtime-transcript" and event.is_final and event.role == "user":
                        source_final.append(event.text)
                        if event.item_id:
                            source_items.add(event.item_id)
                    if event.type == "realtime-text-delta":
                        translated.append(event.text_delta)
                        correlated |= bool(event.provider_metadata.get("previous_item_id"))
                    if event.type == "realtime-audio-output":
                        audio_bytes += len(event.audio)
                    if event.type == "realtime-error":
                        errors = True
                    if event.type == "realtime-end":
                        finished = event.reason == "session.finished"

            async with asyncio.TaskGroup() as group:
                group.create_task(receive_native())
                group.create_task(receive())
                group.create_task(send())
            text = "".join(translated).lower()
            source = " ".join(source_final).lower()
            checks = result["checks"]
            checks.update(source_transcript=bool(source.strip()), translated_text=bool(text.strip()),
                          spanish_marker="siete" in text, source_marker="seven" in source,
                          output_audio=audio_bytes > 0 if audio else audio_bytes == 0,
                          bilingual_correlation=correlated, session_finished=finished, no_errors=not errors)
            checks["source_segments"] = len(source_items) >= min_segments
            result["source_segment_count"] = len(source_items)
            result["status"] = "passed" if all(checks.values()) else "failed"
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["http_status"] = getattr(getattr(error, "response", None), "status_code", None)
    finally:
        if session is not None:
            try:
                await session.aclose()
            except Exception:
                result["status"] = "failed"
    return result


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests/fixtures/realtime/orbit-seven-16k.wav")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--image", type=Path, help="Optional synthetic JPEG for audiovisual input.")
    parser.add_argument("--multi-speaker-fixture", type=Path, help="Optional synthetic two-speaker WAV; requires multiple completed source items.")
    parser.add_argument("--voice-clone", action="store_true", help="Exercise dynamic voice-clone configuration with synthetic speech.")
    parser.add_argument("--region", choices=["intl", "cn"], default="intl")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Choose a fresh evidence path.")
    if not (os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")):
        parser.error("Configure QWEN_API_KEY or DASHSCOPE_API_KEY.")
    artifact = None
    if os.getenv("ZHIVEX_SMOKE_USE_INSTALLED") == "1":
        if args.wheel is None:
            parser.error("Installed smoke requires --wheel.")
        # Reuse the byte-for-byte wheel verifier without importing checkout SDK code.
        from smoke_qwen_omni import verify_installed_wheel
        artifact = verify_installed_wheel(args.wheel)
    pcm = read_pcm(args.fixture)
    multi_pcm = read_pcm(args.multi_speaker_fixture) if args.multi_speaker_fixture else None
    image = args.image.read_bytes() if args.image else None
    provider = create_qwen(region=args.region, base_url=os.getenv("QWEN_BASE_URL"), realtime_url=os.getenv("QWEN_REALTIME_URL"))
    report = {"schema_version": 1, "provider": "qwen", "model": MODEL,
              "recorded_at": datetime.now(timezone.utc).isoformat(), "region": args.region,
              "evidence_status": "integration-only", "artifact": artifact,
              "fixture_sha256": hashlib.sha256(pcm).hexdigest(), "operations": {}}
    operations = report["operations"]
    operations["text_translation"] = await scenario(provider, pcm, audio=False)
    # Authentication/availability failures should not trigger additional billable attempts.
    if operations["text_translation"]["status"] == "passed":
        operations["speech_translation"] = await scenario(provider, pcm, audio=True)
        if image:
            operations["audiovisual_translation"] = await scenario(provider, pcm, audio=False, image=image)
        if args.voice_clone:
            operations["voice_clone_configuration"] = await scenario(provider, pcm, audio=True, clone=True)
        if multi_pcm is not None:
            report["multi_speaker_fixture_sha256"] = hashlib.sha256(multi_pcm).hexdigest()
            operations["multi_speaker_translation"] = await scenario(provider, multi_pcm, audio=True, clone=True, min_segments=2)
    report["status"] = "passed" if all(op["status"] == "passed" for op in operations.values()) else "failed"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
