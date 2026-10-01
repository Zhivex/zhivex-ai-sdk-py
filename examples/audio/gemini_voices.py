"""Create a Beta Gemini voice and synthesize audio; consumes project quota."""
from __future__ import annotations

import asyncio
from pathlib import Path

from zhivex_ai import aclose_default_clients, create_gemini, generate_speech
from zhivex_ai.experimental.gemini import GeminiVoicesClient


async def main() -> None:
    provider = create_gemini()
    voices: GeminiVoicesClient = provider.native.voices()
    voice_id: str | None = None
    try:
        voice = await voices.design(
            prompt="A warm, clear narrator speaking conversational English",
            model="gemini-3.8-flash-tts",
            display_name="SDK example narrator",
        )
        voice_id = voice["id"]
        speech = await generate_speech(
            model=provider.speech_model("gemini-3.8-flash-tts"),
            input="Welcome. This voice was designed from a written description.",
            voice=voice_id,
        )
        Path("gemini-voice.wav").write_bytes(speech.audio)
    finally:
        try:
            if voice_id is not None:
                await voices.delete(voice_id)
        finally:
            await aclose_default_clients()


if __name__ == "__main__":
    asyncio.run(main())
