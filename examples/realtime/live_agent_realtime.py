import asyncio

from zhivex_ai import Agent, create_openai, tool
from zhivex_ai.live import RealtimeSessionConfig, stream_live_agent


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

    async with stream_live_agent(
        agent=agent,
        prompt="What is the weather in Buenos Aires?",
        realtime_config=RealtimeSessionConfig(
            voice="alloy",
            input_audio_media_type="audio/pcm",
            output_audio_media_type="audio/pcm",
        ),
        stream_buffer_size=4096,
    ) as stream:
        async for event in stream.event_stream():
            if getattr(event, "type", "") == "text-delta":
                print(event.text_delta)
        result = await stream.collect()
        print(result.text)


asyncio.run(main())
