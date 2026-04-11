import asyncio

from zhivex_ai import RealtimeSessionConfig, create_azure_openai


async def main() -> None:
    provider = create_azure_openai()
    session = await provider.realtime_model("gpt-realtime").connect(
        RealtimeSessionConfig(
            instructions="Answer in Spanish.",
            voice="alloy",
            input_audio_media_type="audio/pcm",
            output_audio_media_type="audio/pcm",
        )
    )
    await session.send_text("What is a realtime API?")
    async for event in session.event_stream():
        print(event)
        if event.type == "realtime-end":
            break


asyncio.run(main())
