import asyncio

from zhivex_ai import RealtimeSessionConfig, create_bedrock


async def connection_factory(url: str, headers: dict[str, str], options):
    raise RuntimeError(
        "Inject an AWS-signed realtime websocket connection here. "
        "Bedrock realtime support in this SDK is transport-pluggable."
    )


async def main() -> None:
    provider = create_bedrock(region="us-east-1", realtime_connection_factory=connection_factory)
    session = await provider.realtime_model("amazon.nova-sonic-v1").connect(
        RealtimeSessionConfig(
            instructions="Act as a voice assistant.",
            input_audio_media_type="audio/pcm",
            output_audio_media_type="audio/pcm",
        )
    )
    await session.send_text("Hello there.")
    async for event in session.event_stream():
        print(event)
        if event.type == "realtime-end":
            break


asyncio.run(main())
