from __future__ import annotations

import asyncio
from unittest import IsolatedAsyncioTestCase

from zhivex_ai import Agent, ValidationError, stream_agent, stream_object, stream_text
from zhivex_ai._streaming import Broadcast
from zhivex_ai.types import ModelCapabilities, StreamTextDeltaEvent


class WaitingModel:
    provider = "test"
    model_id = "waiting"
    capabilities = ModelCapabilities(
        streaming=True,
        structured_output=True,
        tools=True,
        json_mode=True,
        tool_choice=True,
        parallel_tool_calls=False,
        vision=False,
        files=False,
        audio_input=False,
        audio_output=False,
        embeddings=False,
        reasoning=False,
        web_search=False,
    )

    def __init__(self):
        self.started = asyncio.Event()
        self.closed = asyncio.Event()

    async def stream(self, request):
        async def events():
            try:
                self.started.set()
                yield StreamTextDeltaEvent(text_delta='{"value":')
                await asyncio.Event().wait()
            finally:
                self.closed.set()

        return events()


class StreamLifecycleTests(IsolatedAsyncioTestCase):
    async def test_fastapi_response_close_stops_generation(self):
        import importlib.util
        if importlib.util.find_spec("fastapi") is None:
            self.skipTest("api extra is not installed")
        from examples.integrations.fastapi_streaming_api import _to_fastapi_stream
        from zhivex_ai import to_text_stream_response

        model = WaitingModel()
        result = stream_text(model=model, prompt="test", stream_buffer_size=8)
        response = _to_fastapi_stream(to_text_stream_response(result), result)
        await anext(response.body_iterator)
        await response.body_iterator.aclose()
        self.assertTrue(model.closed.is_set())

    async def test_text_context_cancels_and_joins_provider(self):
        model = WaitingModel()
        async with stream_text(model=model, prompt="test") as result:
            await asyncio.wait_for(model.started.wait(), 1)
        self.assertTrue(model.closed.is_set())
        with self.assertRaises(asyncio.CancelledError):
            await result.collect()
        await result.aclose()

    async def test_object_close_cancels_upstream_even_before_runner_starts(self):
        for started in (False, True):
            model = WaitingModel()
            result = stream_object(model=model, schema=dict, prompt="test")
            if started:
                await asyncio.wait_for(model.started.wait(), 1)
            await result.aclose()
            self.assertTrue(result._upstream._runner.done())
            if started:
                self.assertTrue(model.closed.is_set())

    async def test_agent_close_cancels_upstream(self):
        model = WaitingModel()
        result = stream_agent(agent=Agent(name="test", model=model), prompt="test")
        await asyncio.wait_for(model.started.wait(), 1)
        await result.aclose()
        self.assertTrue(model.closed.is_set())

    async def test_closing_one_subscriber_does_not_cancel_other_subscribers(self):
        model = WaitingModel()
        result = stream_text(model=model, prompt="test")
        first, second = result.event_stream(), result.event_stream()
        try:
            self.assertEqual(await anext(first), await anext(second))
            await first.aclose()
            self.assertFalse(model.closed.is_set())
        finally:
            await second.aclose()
            await result.aclose()

    async def test_slow_and_late_consumers_fail_explicitly_without_unbounded_queues(
        self,
    ):
        broadcast = Broadcast[int](max_events=2)
        await broadcast.publish(0)
        slow = broadcast.stream()
        self.assertEqual(await anext(slow), 0)
        for item in range(1, 10000):
            await broadcast.publish(item)
        self.assertEqual(len(broadcast.history), 2)
        for consumer in (slow, broadcast.stream()):
            with self.assertRaisesRegex(ValidationError, "retained event history"):
                await anext(consumer)

    async def test_active_consumers_receive_every_event_and_close_wakes_waiters(self):
        broadcast = Broadcast[int](max_events=2)
        first, second = broadcast.stream(), broadcast.stream()
        for item in range(20):
            pending = [
                asyncio.create_task(anext(first)),
                asyncio.create_task(anext(second)),
            ]
            await asyncio.sleep(0)
            await broadcast.publish(item)
            self.assertEqual(await asyncio.gather(*pending), [item, item])
        pending_end = asyncio.create_task(anext(first))
        await asyncio.sleep(0)
        await broadcast.close()
        with self.assertRaises(StopAsyncIteration):
            await asyncio.wait_for(pending_end, 1)
        await second.aclose()

    async def test_immediate_close_wakes_event_consumers(self):
        result = stream_text(model=WaitingModel(), prompt="test")
        await result.aclose()
        self.assertEqual([event async for event in result.event_stream()], [])

    async def test_full_replay_opt_in_and_invalid_limits(self):
        broadcast = Broadcast[int](max_events=None)
        for item in range(5000):
            await broadcast.publish(item)
        await broadcast.close()
        self.assertEqual(
            [event async for event in broadcast.stream()], list(range(5000))
        )
        for limit in (0, -1, True, 1.5):
            with self.assertRaises(ValidationError):
                stream_text(
                    model=WaitingModel(), prompt="test", stream_buffer_size=limit
                )
