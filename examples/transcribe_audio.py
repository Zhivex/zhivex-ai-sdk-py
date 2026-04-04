import asyncio
from pathlib import Path

from zhivex_ai import AudioInput, create_openai, transcribe_audio


async def main() -> None:
    provider = create_openai()
    audio_path = Path("sample.wav")
    audio = AudioInput(
        data=audio_path.read_bytes(),
        media_type="audio/wav",
        filename=audio_path.name,
    )

    result = await transcribe_audio(
        model=provider.transcription_model("gpt-4o-mini-transcribe"),
        audio=audio,
    )
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
