from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch
import wave

from scripts.smoke_qwen_translate import read_pcm, scenario
from zhivex_ai import create_qwen
from tests.test_qwen_realtime import Connection


class TranslateSmokeTests(IsolatedAsyncioTestCase):
    async def test_smoke_validates_evidence_and_omits_content(self):
        connection = Connection()
        connection.final_events = [
            {"type": "conversation.item.created", "item": {"id": "translated"}, "previous_item_id": "source"},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "orbit seven", "item_id": "source"},
            {"type": "response.text.delta", "delta": "órbita siete", "item_id": "translated"},
            {"type": "response.done", "response": {"status": "completed"}},
        ]

        async def factory(*_):
            return connection

        provider = create_qwen(api_key="secret", realtime_connection_factory=factory)
        with patch("scripts.smoke_qwen_translate.asyncio.sleep", return_value=None):
            report = await scenario(provider, b"\0\0", audio=False)
        self.assertEqual(report["status"], "passed")
        self.assertNotIn("órbita", str(report))
        self.assertNotIn("secret", str(report))

    async def test_missing_translation_never_passes(self):
        connection = Connection()

        async def factory(*_):
            return connection

        with patch("scripts.smoke_qwen_translate.asyncio.sleep", return_value=None):
            report = await scenario(create_qwen(api_key="secret", realtime_connection_factory=factory), b"\0\0", audio=True)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["checks"]["translated_text"])

    async def test_transport_error_is_redacted(self):
        async def factory(*_):
            raise RuntimeError("private credential")

        report = await scenario(create_qwen(api_key="secret", realtime_connection_factory=factory), b"\0\0", audio=False)
        self.assertEqual(report["error_type"], "RuntimeError")
        self.assertNotIn("private", str(report))
        self.assertEqual(report["status"], "failed")


class TranslateFixtureTests(TestCase):
    def test_wav_header_is_removed_and_rate_is_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.wav"
            for rate in (16000, 24000):
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(rate)
                    wav.writeframes(b"\0\0")
                if rate == 16000:
                    self.assertEqual(read_pcm(path), b"\0\0")
                else:
                    with self.assertRaises(ValueError):
                        read_pcm(path)
