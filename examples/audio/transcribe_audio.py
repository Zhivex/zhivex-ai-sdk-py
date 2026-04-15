import asyncio
from pathlib import Path

from zhivex_ai import AudioInput, create_openai, transcribe_audio


async def main() -> None:
    provider = create_openai()
    audio_path = Path(__file__).with_name("speech.wav")
    if not audio_path.exists():
        raise SystemExit(f"Add a WAV file at {audio_path} before running this example.")
    audio = AudioInput(
        data=audio_path.read_bytes(),
        media_type="audio/wav",
        filename=audio_path.name,
    )

    result = await transcribe_audio(
        model=provider.transcription_model("gpt-4o-transcribe"),
        audio=audio,
    )
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
