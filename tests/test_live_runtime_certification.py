from __future__ import annotations

import asyncio
import json
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from scripts import certify_live_runtime


class LiveCertificationTests(IsolatedAsyncioTestCase):
    async def test_recovery_requires_explicit_text_model(self) -> None:
        result = await certify_live_runtime.probe("gemini", "test", "approval-allow-restart")
        self.assertEqual(result, {"status": "blocked", "reason": "missing_resume_model"})

    async def test_audio_roundtrip_requires_spoken_marker_without_text_prompt(self) -> None:
        class Connection:
            def __init__(self):
                self.queue = asyncio.Queue()
                self.sent = []

            async def send_json(self, payload):
                self.sent.append(payload)
                if payload.get("type") == "response.create":
                    await self.queue.put({"type": "response.output_audio.delta", "delta": "AQI="})
                    await self.queue.put({"type": "response.output_audio_transcript.done", "transcript": "Orbit seven"})
                    await self.queue.put({"type": "response.done", "response": {"status": "completed"}})

            async def recv_json(self):
                return await self.queue.get()

            async def close(self):
                pass

        connection = Connection()

        async def connect(*args, **kwargs):
            return connection

        with patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic-test-key"}), patch.object(
            certify_live_runtime, "open_websocket_connection", connect,
        ):
            result = await certify_live_runtime.probe("openai", "test-model", "audio-roundtrip")
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["marker_matched"])
        self.assertGreater(result["input_audio_bytes"], 4800)
        self.assertNotIn("conversation.item.create", [p["type"] for p in connection.sent])
        self.assertNotIn("orbit", json.dumps(connection.sent).lower())

    async def test_audio_evidence_counts_normalized_audio_and_closes_connection(self) -> None:
        class Connection:
            def __init__(self):
                self.queue = asyncio.Queue()
                self.closed = False

            async def send_json(self, payload):
                if payload.get("type") == "response.create":
                    await self.queue.put({"type": "response.output_audio.delta", "delta": "AQI="})
                    await self.queue.put({"type": "response.done", "response": {"status": "completed"}})

            async def recv_json(self):
                return await self.queue.get()

            async def close(self):
                self.closed = True

        connection = Connection()

        async def connect(*args, **kwargs):
            return connection

        with patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic-test-key"}), patch.object(
            certify_live_runtime, "open_websocket_connection", connect,
        ):
            result = await certify_live_runtime.probe("openai", "test-model", "audio-output")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["audio_bytes"], 2)
        self.assertTrue(connection.closed)
        self.assertNotIn("synthetic-test-key", json.dumps(result))

    async def test_missing_credentials_are_blocked_without_connecting(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch.object(
            certify_live_runtime, "open_websocket_connection",
        ) as connection:
            result = await certify_live_runtime.probe("openai", "test-model", "response")
        self.assertEqual(result, {"status": "blocked", "reason": "missing_credentials"})
        connection.assert_not_called()

    async def test_transient_connection_closure_is_retried(self) -> None:
        class ConnectionClosedError(Exception):
            pass

        class Connection:
            def __init__(self, *, fail: bool):
                self.fail = fail
                self.queue = asyncio.Queue()

            async def send_json(self, payload):
                if not self.fail and payload.get("type") == "response.create":
                    await self.queue.put({"type": "response.text.delta", "delta": "ok"})
                    await self.queue.put({"type": "response.done", "response": {"status": "completed"}})

            async def recv_json(self):
                if self.fail:
                    raise ConnectionClosedError("transient close")
                return await self.queue.get()

            async def close(self):
                pass

        attempts = 0

        async def connect(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            return Connection(fail=attempts == 1)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic-test-key"}), patch.object(
            certify_live_runtime, "open_websocket_connection", connect,
        ):
            result = await certify_live_runtime.probe("openai", "test-model", "response")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(attempts, 2)
