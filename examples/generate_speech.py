import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from zhivex_ai import create_gemini, generate_speech

load_dotenv()


async def main() -> None:
    provider = create_gemini(api_key=os.getenv("GOOGLE_API_KEY"))
    result = await generate_speech(
        model=provider.speech_model("gemini-2.5-flash-preview-tts"),
        input="Zhivex AI SDK makes provider switching easier.",
        voice="Kora",
    )

    Path("speech.mp3").write_bytes(result.audio)
    print("saved speech.mp3", result.media_type)


if __name__ == "__main__":
    asyncio.run(main())
