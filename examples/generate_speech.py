import asyncio
from pathlib import Path

from zhivex_ai import create_openai, generate_speech


async def main() -> None:
    provider = create_openai()
    result = await generate_speech(
        model=provider.speech_model("gpt-4o-mini-tts"),
        input="Zhivex AI SDK makes provider switching easier.",
        voice="alloy",
    )

    Path("speech.mp3").write_bytes(result.audio)
    print("saved speech.mp3", result.media_type)


if __name__ == "__main__":
    asyncio.run(main())
