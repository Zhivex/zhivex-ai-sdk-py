"""GPT-Live voice with an application-owned durable Agent backend.

Run with OPENAI_API_KEY, a mono PCM16/24kHz WAV input and output WAV path.
The output file records received audio; this headless example does not play it.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
import base64
from pathlib import Path
import wave

from zhivex_ai import Agent, RunLimits, create_openai, create_sqlite_agent_run_store
from zhivex_ai.live import RealtimeConnectOptions


async def main(input_path: Path, output_path: Path, database: Path) -> None:
    provider = create_openai()
    agent = Agent(
        name="voice-backend", model=provider("gpt-5.6-luna"),
        instructions="Answer the user's latest request using the conversation context. Return under 400 UTF-8 bytes.",
        run_store=create_sqlite_agent_run_store(str(database)),
        run_limits=RunLimits(max_steps=3, max_wall_time_ms=20000),
    )
    with wave.open(str(input_path)) as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 24000):
            raise ValueError("Expected mono PCM16 at 24 kHz")
        pcm = audio.readframes(audio.getnframes())
    session = await provider.native.live().connect({
        "model": "gpt-live-1",
        "instructions": "Be concise. Delegate questions requiring reasoning to the backend.",
        "delegation": {"type": "client"},
        "audio": {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": "marin"}},
    }, RealtimeConnectOptions(timeout_ms=10000))
    history = []
    seen = set()
    tasks = []

    async def feed():
        for offset in range(0, len(pcm), 4800):
            await session.append_audio(pcm[offset:offset + 4800])
            await asyncio.sleep(.1)
        while True:
            await session.append_audio(bytes(4800))
            await asyncio.sleep(.1)

    tasks.append(asyncio.create_task(feed()))
    try:
        with wave.open(str(output_path), "wb") as output:
            output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            async with asyncio.timeout(45):
                while True:
                    for task in tasks:
                        if task.done():
                            task.result()  # Surface failures; do not silently lose backend errors.
                    event = await session.receive()
                    kind = event.get("type")
                    if kind == "error":
                        raise RuntimeError("GPT-Live provider error; inspect application diagnostics")
                    if kind in {"session.input_transcript.delta", "session.output_transcript.delta"}:
                        history.append({"speaker": "user" if kind == "session.input_transcript.delta" else "assistant",
                                        "text": event["delta"], "start_ms": event.get("start_ms"), "end_ms": event.get("end_ms")})
                    if kind == "session.output_audio.delta":
                        output.writeframes(base64.b64decode(event["delta"], validate=True))
                    if kind == "session.delegation.created" and event["delegation"]["target"] == "client":
                        identifier = event["delegation"]["id"]
                        if identifier not in seen:
                            seen.add(identifier)
                            # Context selection is an application policy. Keep receiving voice
                            # while the Agent runs; interruption does not imply tool cancellation.
                            prompt = "Conversation fragments (may contain corrections):\n" + repr(history)
                            tasks.append(asyncio.create_task(session.run_delegation(
                                event, agent=agent, prompt=prompt, max_tokens=128,
                                timeout_ms=15000, max_retries=0,
                            )))
    except TimeoutError:
        pass  # This demo has an explicit 45-second conversation budget.
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            final = await session.finish(timeout_ms=10000)
            print("Session finalized; usage present:", "usage" in final)
        finally:
            with suppress(Exception):
                await session.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--database", type=Path, default=Path("gpt-live-runs.sqlite"))
    args = parser.parse_args()
    asyncio.run(main(args.input, args.output, args.database))
