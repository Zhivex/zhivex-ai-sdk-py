import asyncio
import os
from pathlib import Path
import wave

from _bootstrap import load_dotenv_if_available

from zhivex_ai import create_gemini, generate_speech

load_dotenv_if_available()


def save_wave(path: Path, pcm: bytes, *, channels: int = 1, rate: int = 24_000, sample_width: int = 2) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm)


async def main() -> None:
    provider = create_gemini(api_key=os.getenv("GOOGLE_API_KEY"))
    result = await generate_speech(
        model=provider.speech_model("gemini-2.5-flash-preview-tts"),
        input="Zhivex AI SDK makes provider switching easier.",
        voice="Kore",
    )

    output_path = Path("speech.wav")
    save_wave(output_path, result.audio)
    print(f"saved {output_path}", result.media_type)


if __name__ == "__main__":
    asyncio.run(main())
