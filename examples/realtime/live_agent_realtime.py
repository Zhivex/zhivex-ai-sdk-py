import asyncio

from zhivex_ai import Agent, create_openai, tool
from zhivex_ai.experimental import RealtimeSessionConfig, stream_live_agent


async def main() -> None:
    provider = create_openai()
    agent = Agent(
        name="voice-assistant",
        instructions="Be brief and helpful.",
        model=provider.realtime_model("gpt-realtime-2.1"),
        tools={
            "lookup_weather": tool(
                name="lookup_weather",
                description="Returns a mock weather forecast.",
                schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                execute=lambda payload: {"forecast": f"Sunny in {payload['city']}"},
            )
        },
    )

    stream = stream_live_agent(
        agent=agent,
        realtime_config=RealtimeSessionConfig(
            voice="alloy",
            input_audio_media_type="audio/pcm",
            output_audio_media_type="audio/pcm",
        ),
    )
    await stream.send_text("What is the weather in Buenos Aires?")
    async for event in stream.event_stream():
        print(event)
        if getattr(event, "type", "") == "finish":
            break


asyncio.run(main())
