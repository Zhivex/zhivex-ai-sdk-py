import asyncio

from zhivex_ai import create_openai
from zhivex_ai.experimental import (
    RealtimeAudioOutputEvent,
    RealtimeResponseCompletedEvent,
    RealtimeSessionConfig,
    RealtimeSessionEndedEvent,
    RealtimeTextDeltaEvent,
    RealtimeTranscriptEvent,
)


async def main() -> None:
    provider = create_openai()
    session = await provider.realtime_model("gpt-realtime-2.1").connect(
        RealtimeSessionConfig(
            instructions="Be concise.",
            voice="alloy",
            input_audio_media_type="audio/pcm",
            output_audio_media_type="audio/pcm",
        )
    )
    try:
        await session.send_text("Say hello in one short sentence.")
        audio_bytes = 0
        printed_text = False
        async for event in session.event_stream():
            if isinstance(event, RealtimeTextDeltaEvent):
                print(event.text_delta, end="", flush=True)
                printed_text = True
                continue
            if isinstance(event, RealtimeTranscriptEvent) and event.role == "assistant" and event.is_final and event.text:
                if printed_text:
                    print()
                    printed_text = False
                print(f"Assistant transcript: {event.text}")
                continue
            if isinstance(event, RealtimeAudioOutputEvent):
                audio_bytes += len(event.audio)
                continue
            if isinstance(event, RealtimeResponseCompletedEvent):
                if printed_text:
                    print()
                if audio_bytes:
                    print(f"[audio chunks received: {audio_bytes} bytes pcm]")
                print(f"[turn complete: {event.reason}]")
                break
            if isinstance(event, RealtimeSessionEndedEvent):
                if printed_text:
                    print()
                print(f"[session ended: {event.reason}]")
                break
    finally:
        await session.aclose()


asyncio.run(main())
