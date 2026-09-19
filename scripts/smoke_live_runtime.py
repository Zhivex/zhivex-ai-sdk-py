"""Offline acceptance smoke for the installed Stable live runtime (no provider calls)."""
from __future__ import annotations

import asyncio
from typing import Any

from zhivex_ai import Agent, ModelCapabilities, create_in_memory_agent_run_store
from zhivex_ai.live import RealtimeResponseCompletedEvent, RealtimeTextDeltaEvent, stream_live_agent


class Session:
    def __init__(self, *, complete: bool) -> None:
        self.complete = complete
        self.closed = False

    async def send_text(self, text: str) -> None:
        pass

    async def aclose(self) -> None:
        self.closed = True

    def event_stream(self):
        async def events():
            yield RealtimeTextDeltaEvent(text_delta="installed-live-ok")
            if self.complete:
                yield RealtimeResponseCompletedEvent(reason="done")
        return events()


class Model:
    provider = "test"
    model_id = "offline-live-contract"
    capabilities = ModelCapabilities(
        streaming=False, tools=True, structured_output=False, json_mode=False,
        tool_choice=False, parallel_tool_calls=False, vision=False, files=False,
        audio_input=True, audio_output=True, embeddings=False, reasoning=False,
        web_search=False, realtime=True,
    )

    def __init__(self, session: Session) -> None:
        self.session = session

    async def connect(self, config: Any = None, options: Any = None) -> Session:
        return self.session


async def main() -> None:
    for complete in (True, False):
        session = Session(complete=complete)
        store = create_in_memory_agent_run_store()
        agent = Agent(name="installed-live", model=Model(session), run_store=store)
        async with stream_live_agent(agent=agent, idempotency_key="smoke", stream_buffer_size=32) as stream:
            if complete:
                result = await stream.collect()
                assert result.text == "installed-live-ok"
            else:
                try:
                    await stream.collect()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Truncated realtime input was accepted as success")
        state = await store.find_by_idempotency_key("smoke")
        assert state is not None and state.status == ("completed" if complete else "failed")
        assert session.closed
    print("Installed Stable live runtime: completion, truncation, durable state and cleanup passed.")


if __name__ == "__main__":
    asyncio.run(main())
