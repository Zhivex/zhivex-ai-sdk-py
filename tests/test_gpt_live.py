from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from zhivex_ai import ValidationError, create_openai
from zhivex_ai.providers._native_sessions import OpenAILiveSession


class Connection:
    def __init__(self):
        self.sent = []
        self.incoming = asyncio.Queue()
        self.closed = False

    async def send_json(self, event):
        self.sent.append(event)

    async def recv_json(self):
        return await self.incoming.get()

    async def close(self):
        self.closed = True


class GPTLiveTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.connection = Connection()
        self.session = OpenAILiveSession(self.connection, {"session": {"id": "voice-1"}})
        self.event = {"type": "session.delegation.created", "delegation": {"id": "task-1", "target": "client"}}
        self.agent = SimpleNamespace(run_store=object())

    async def test_audio_and_context_wire_validation(self):
        await self.session.append_audio(b'\x00\x01')
        self.assertEqual(self.connection.sent[0], {"type": "session.input_audio.append", "audio": "AAE="})
        for invalid in (b'', b'x', 'abc'):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                await self.session.append_audio(invalid)
        await self.session.append_context("Ready", event_id="ready", delegation_id="task-1")
        self.assertEqual(self.connection.sent[-1]["delegation_id"], "task-1")
        for content in ('', 'é' * 251):
            with self.assertRaises(ValidationError):
                await self.session.append_context(content, event_id="bad")

    async def test_one_reader_and_finish_refuses_competing_reader(self):
        reader = asyncio.create_task(self.session.receive())
        await asyncio.sleep(0)
        with self.assertRaises(ValidationError):
            await self.session.receive()
        with self.assertRaises(ValidationError):
            await self.session.finish()
        self.assertFalse(self.session.closed)
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        await self.connection.incoming.put({"type": "session.closed", "usage": {}})
        await self.session.finish()
        self.assertTrue(self.connection.closed)
        with self.assertRaises(ValidationError):
            await self.session.receive()

    async def test_delegation_is_claimed_before_backend_await(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        result = SimpleNamespace(state=SimpleNamespace(status="completed"), text="ORBIT_SEVEN")

        async def run(**kwargs):
            self.assertEqual(kwargs['idempotency_key'], 'gpt-live:voice-1:task-1')
            self.assertEqual(kwargs['prompt'], 'application context')
            entered.set()
            await release.wait()
            return result

        with patch('zhivex_ai.agent.run_agent', side_effect=run):
            task = asyncio.create_task(self.session.run_delegation(self.event, agent=self.agent, prompt='application context'))
            await entered.wait()
            with self.assertRaises(ValidationError):
                await self.session.run_delegation(self.event, agent=self.agent, prompt='duplicate')
            release.set()
            self.assertIs(await task, result)
        self.assertEqual(len(self.connection.sent), 1)
        self.assertEqual(self.connection.sent[0]['content'], 'ORBIT_SEVEN')

    async def test_suspended_or_failed_backend_never_announces_completion(self):
        for state in ('suspended', 'failed', 'cancelled'):
            with self.subTest(state=state):
                session = OpenAILiveSession(self.connection, {'session': {'id': 'voice-1'}})
                run = AsyncMock(return_value=SimpleNamespace(state=SimpleNamespace(status=state), text='unverified'))
                with patch('zhivex_ai.agent.run_agent', run):
                    await session.run_delegation(self.event, agent=self.agent, prompt='context')
        self.assertEqual(self.connection.sent, [])

    async def test_failed_delegation_is_not_reexecuted(self):
        run = AsyncMock(side_effect=RuntimeError('backend failed'))
        with patch('zhivex_ai.agent.run_agent', run):
            with self.assertRaises(RuntimeError):
                await self.session.run_delegation(self.event, agent=self.agent, prompt='context')
            with self.assertRaises(ValidationError):
                await self.session.run_delegation(self.event, agent=self.agent, prompt='context')
        self.assertEqual(run.call_count, 1)

    async def test_cancelled_backend_claim_is_not_replayed(self):
        entered = asyncio.Event()

        async def run(**_):
            entered.set()
            await asyncio.Event().wait()

        with patch('zhivex_ai.agent.run_agent', side_effect=run):
            task = asyncio.create_task(self.session.run_delegation(self.event, agent=self.agent, prompt='context'))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            with self.assertRaises(ValidationError):
                await self.session.run_delegation(self.event, agent=self.agent, prompt='context')
        self.assertEqual(self.connection.sent, [])

    async def test_startup_error_survives_cleanup_error(self):
        async def factory(*_):
            self.connection.close = AsyncMock(side_effect=RuntimeError('cleanup'))
            await self.connection.incoming.put({'type': 'error'})
            return self.connection
        from zhivex_ai import ParseError
        with self.assertRaises(ParseError):
            await create_openai(api_key='test', realtime_connection_factory=factory).native.live().connect({'model': 'gpt-live-1'})
