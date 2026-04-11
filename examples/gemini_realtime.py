import asyncio

from zhivex_ai import RealtimeSessionConfig, create_gemini


async def main() -> None:
    provider = create_gemini()
    session = await provider.realtime_model("gemini-live-2.5-flash").connect(
        RealtimeSessionConfig(
            instructions="Keep responses short.",
            output_audio_media_type="audio/pcm",
        )
    )
    await session.send_text("Explain what low latency means for voice assistants.")
    async for event in session.event_stream():
        print(event)
        if event.type == "realtime-end":
            break


asyncio.run(main())
