"""Translate a local mono PCM16/16 kHz WAV, optionally with JPEG context (Beta)."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
from typing import cast
import wave

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))
from _bootstrap import load_dotenv_if_available

load_dotenv_if_available()

from zhivex_ai import AudioFrame, create_qwen
from zhivex_ai.experimental.qwen import QwenLiveTranslateSession
from zhivex_ai.live import RealtimeSessionConfig


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--language", default="es")
    parser.add_argument("--region", choices=["intl", "cn"], default="intl")
    parser.add_argument("--output", type=Path, help="Save translated PCM16/24 kHz speech as WAV; otherwise text only.")
    parser.add_argument("--image", type=Path, help="Optional JPEG context, max 500000 bytes and 1080p.")
    parser.add_argument("--clone-voice", action="store_true", help="Clone the synthetic/consented source speaker; requires --output.")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Choose a new output path.")
    if args.clone_voice and not args.output:
        parser.error("--clone-voice requires --output.")
    options = {"output_modalities": ["text", "audio"] if args.output else ["text"]}
    if args.clone_voice:
        options.update(enable_voice_clone=True, voice_clone_options={"frequency": "always"})
    provider = create_qwen(region=args.region)
    with wave.open(str(args.input), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (1, 2, 16000, "NONE"):
            parser.error("Input must be mono PCM16 WAV at 16000 Hz.")
        session = cast(QwenLiveTranslateSession, await provider.native.realtime_model(
            "qwen3.8-livetranslate-flash-realtime"
        ).connect(RealtimeSessionConfig(translation_target_language_code=args.language, provider_options=options)))
        output = wave.open(str(args.output), "wb") if args.output else None
        if output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24000)

        async def send() -> None:
            first = True
            while chunk := source.readframes(1600):
                await session.send_audio(AudioFrame(chunk, "audio/pcm", sample_rate_hz=16000))
                if first and args.image:
                    await session.send_image(args.image.read_bytes())
                first = False
                await asyncio.sleep(0.1)
            await session.finish()

        async def receive() -> None:
            async for event in session.event_stream():
                if event.type == "realtime-transcript" and event.is_final:
                    print(f"{event.role}: {event.text}")
                elif event.type == "realtime-audio-output" and output:
                    output.writeframesraw(event.audio)
                elif event.type == "realtime-error":
                    raise RuntimeError(event.message)

        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(receive())
                group.create_task(send())
        finally:
            try:
                await session.aclose()
            finally:
                if output:
                    output.close()


if __name__ == "__main__":
    asyncio.run(main())
