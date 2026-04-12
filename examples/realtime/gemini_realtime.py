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
        session = await provider.realtime_model("gemini-3.1-flash-live-preview").connect(
            RealtimeSessionConfig(
                instructions="Keep responses short.",
                output_audio_media_type="audio/pcm",
            )
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    await session.send_text("Explain what low latency means for voice assistants.")
    async for event in session.event_stream():
        print(event)
        if event.type == "realtime-end":
            break


asyncio.run(main())
