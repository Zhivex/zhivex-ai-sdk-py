from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import wave
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from _bootstrap import load_dotenv_if_available

load_dotenv_if_available()

from zhivex_ai import (
    RealtimeAudioOutputEvent,
    RealtimeResponseCompletedEvent,
    RealtimeSessionConfig,
    RealtimeSessionEndedEvent,
    RealtimeTextDeltaEvent,
    RealtimeTranscriptEvent,
    create_gemini,
)

GEMINI_LIVE_OUTPUT_SAMPLE_RATE_HZ = 24_000
PCM_SAMPLE_WIDTH_BYTES = 2


def _write_pcm_wav(chunks: list[bytes]) -> Path:
    with tempfile.NamedTemporaryFile(prefix="gemini-live-", suffix=".wav", delete=False) as handle:
        path = Path(handle.name)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(GEMINI_LIVE_OUTPUT_SAMPLE_RATE_HZ)
        wav_file.writeframes(b"".join(chunks))
    return path


async def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit('Missing GOOGLE_API_KEY. Export it or put it in a local ".env" file.')

    provider = create_gemini(api_key=api_key)
    try:
        session = await provider.realtime_model("gemini-2.5-flash-native-audio-preview-12-2025").connect(
            RealtimeSessionConfig(
                instructions="Keep responses short.",
                output_audio_media_type="audio/pcm",
                output_sample_rate_hz=GEMINI_LIVE_OUTPUT_SAMPLE_RATE_HZ,
            )
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    try:
        await session.send_text("Explain what low latency means for voice assistants.")
        audio_bytes = 0
        audio_chunks: list[bytes] = []
        printed_text = False
        async for event in session.event_stream():
            if isinstance(event, RealtimeTextDeltaEvent):
                print(event.text_delta, end="", flush=True)
                printed_text = True
                continue
            if isinstance(event, RealtimeTranscriptEvent) and event.role == "assistant" and event.is_final and event.text:
                if printed_text:
                    print()
                    printed_text = False
                print(f"Assistant transcript: {event.text}")
                continue
            if isinstance(event, RealtimeAudioOutputEvent):
                audio_bytes += len(event.audio)
                audio_chunks.append(event.audio)
                continue
            if isinstance(event, RealtimeResponseCompletedEvent):
                if printed_text:
                    print()
                if audio_bytes:
                    print(f"[audio chunks received: {audio_bytes} bytes pcm]")
                    print(f"[wav saved: {_write_pcm_wav(audio_chunks)}]")
                print(f"[turn complete: {event.reason}]")
                break
            if isinstance(event, RealtimeSessionEndedEvent):
                if printed_text:
                    print()
                print(f"[session ended: {event.reason}]")
                break
    finally:
        await session.aclose()


asyncio.run(main())
