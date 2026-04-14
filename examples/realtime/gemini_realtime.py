from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from _bootstrap import load_dotenv_if_available

load_dotenv_if_available()

from zhivex_ai import RealtimeSessionConfig, create_gemini


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
            )
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    try:
        await session.send_text("Explain what low latency means for voice assistants.")
        async for event in session.event_stream():
            print(event)
            if event.type in {"realtime-response-complete", "realtime-end"}:
                break
    finally:
        await session.aclose()


asyncio.run(main())
