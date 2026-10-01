from __future__ import annotations

import asyncio
import base64
from unittest import IsolatedAsyncioTestCase, TestCase

from zhivex_ai import AudioFrame, ConfigurationError, UnsupportedFeatureError, ValidationError, create_qwen
from zhivex_ai.experimental.qwen import QwenLiveTranslateModel, QwenLiveTranslateSession
from zhivex_ai.live import RealtimeConnectOptions, RealtimeSessionConfig
from zhivex_ai.providers._qwen_realtime import MODEL_ID, realtime_url, session_payload


class Connection:
    def __init__(self, *, ack=True, finish=True):
        self.queue = asyncio.Queue()
        self.queue.put_nowait({"type": "session.created", "session": {"id": "session-1"}})
        self.sent = []
        self.closed = False
        self.ack = ack
        self.finish = finish
        self.final_events = []

    async def send_json(self, payload):
        self.sent.append(payload)
        if payload["type"] == "session.update" and self.ack:
            await self.queue.put({"type": "session.updated", "session": {"id": "session-1", **payload["session"]}})
        if payload["type"] == "session.finish" and self.finish:
            for event in self.final_events:
                await self.queue.put(event)
            await self.queue.put({"type": "session.finished"})

    async def recv_json(self):
        return await self.queue.get()

    async def close(self):
        self.closed = True


class QwenRealtimeTests(IsolatedAsyncioTestCase):
    async def model(self, connection=None, **kwargs):
        connection = connection or Connection()
        self.connection = connection
        self.calls = []

        async def factory(url, headers, options):
            self.calls.append((url, headers, options))
            return connection

        return create_qwen(api_key="not-a-real-key", realtime_connection_factory=factory, **kwargs).native.realtime_model(MODEL_ID)

    async def test_native_factory_endpoint_and_handshake(self):
        model = await self.model(workspace_id="my-workspace")
        self.assertIsInstance(model, QwenLiveTranslateModel)
        self.assertTrue(model.capabilities.realtime)
        self.assertFalse(model.capabilities.realtime_tools)
        session = await model.connect(RealtimeSessionConfig(translation_target_language_code="es"))
        self.assertIsInstance(session, QwenLiveTranslateSession)
        self.assertEqual(self.calls[0][0], f"wss://my-workspace.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime?model={MODEL_ID}")
        self.assertEqual(self.calls[0][1], {"Authorization": "Bearer not-a-real-key"})
        self.assertEqual(self.connection.sent[0]["session"], {"translation": {"language": "es"}, "output_modalities": ["text", "audio"], "audio": {"output": {"voice": "Tina"}}})
        await session.aclose()
        events = [event async for event in session.event_stream()]
        self.assertEqual(events[0].session_id, "session-1")
        self.assertEqual(events[-1].reason, "session.finished")
        self.assertTrue(self.connection.closed)

    async def test_bilingual_events_correlation_audio_usage_and_drain(self):
        model = await self.model()
        session = await model.connect()
        self.connection.final_events = [
            {"type": "conversation.item.created", "previous_item_id": "source", "item": {"id": "translation"}},
            {"type": "conversation.item.input_audio_transcription.delta", "item_id": "source", "delta": "Hello"},
            {"type": "conversation.item.input_audio_transcription.delta", "item_id": "source", "delta": " world"},
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": "source", "transcript": "Hello world", "speaker_id": "speaker-2"},
            {"type": "response.audio_transcript.delta", "item_id": "translation", "response_id": "r", "delta": "Hola"},
            {"type": "response.audio.delta", "item_id": "translation", "response_id": "r", "delta": "AAA="},
            {"type": "response.audio_transcript.done", "item_id": "translation", "transcript": "Hola"},
            {"type": "response.done", "response": {"status": "completed", "usage": {"total_tokens": 5}}},
        ]
        await session.send_audio(AudioFrame(data=b"\0\0", media_type="audio/pcm", is_final=True))
        events = [event async for event in session.event_stream()]
        self.assertEqual([e.text for e in events if e.type == "realtime-transcript" and e.role == "user"], ["Hello", "Hello world", "Hello world"])
        delta = next(e for e in events if e.type == "realtime-text-delta")
        self.assertEqual(delta.provider_metadata["previous_item_id"], "source")
        self.assertEqual(delta.response_id, "r")
        audio = next(e for e in events if e.type == "realtime-audio-output")
        self.assertEqual((audio.audio, audio.sample_rate_hz, audio.channels), (b"\0\0", 24000, 1))
        final = next(e for e in events if e.type == "realtime-response-complete")
        self.assertEqual(final.provider_metadata["response"]["usage"]["total_tokens"], 5)
        source = next(e for e in events if e.type == "realtime-transcript" and e.is_final and e.role == "user")
        self.assertEqual(source.provider_metadata["speaker_id"], "speaker-2")
        raw = [e async for e in session.native_event_stream()]
        self.assertEqual(raw[-1]["type"], "session.finished")
        self.assertEqual(self.connection.sent[-1]["type"], "session.finish")
        await session.aclose()
        self.assertEqual(sum(e["type"] == "session.finish" for e in self.connection.sent), 1)
        with self.assertRaises(ValidationError):
            await session.send_audio(AudioFrame(data=b"\0\0", media_type="audio/pcm"))

    async def test_text_output_and_update_ack(self):
        session = await (await self.model()).connect(RealtimeSessionConfig(provider_options={"output_modalities": ["text"]}))
        await session.update(translation_target_language_code="el", provider_options={"output_modalities": ["text"], "translation": {"corpus": {"phrases": {"AI": "IA"}}}})
        self.assertEqual(session.config.translation_target_language_code, "el")
        self.connection.final_events = [{"type": "response.text.delta", "delta": "Hola"}, {"type": "response.text.done", "text": "Hola"}]
        await session.finish()
        events = [e async for e in session.event_stream()]
        self.assertEqual(events[1].text_delta, "Hola")
        self.assertEqual(events[2].text, "Hola")

    async def test_setup_timeout_closes_transport(self):
        model = await self.model(Connection(ack=False))
        with self.assertRaises(TimeoutError):
            await model.connect(options=RealtimeConnectOptions(timeout_ms=10))
        self.assertTrue(self.connection.closed)

    async def test_setup_error_and_unexpected_ack_close_transport(self):
        for payload in ({"type": "error", "error": {"message": "secret"}}, {"type": "response.created"}, {"code": "Denied", "type": "session.created"}):
            connection = Connection()
            connection.queue.get_nowait()
            connection.queue.put_nowait(payload)
            model = await self.model(connection)
            with self.assertRaises(ConnectionError) as error:
                await model.connect()
            self.assertNotIn("secret", str(error.exception))
            self.assertTrue(connection.closed)

    async def test_finish_timeout_fails_instead_of_false_success(self):
        session = await (await self.model(Connection(finish=False))).connect(options=RealtimeConnectOptions(timeout_ms=10))
        with self.assertRaises(TimeoutError):
            await session.finish()
        self.assertTrue(self.connection.closed)
        self.assertFalse(any(e.reason == "session.finished" for e in [e async for e in session.event_stream()] if e.type == "realtime-end"))

    async def test_update_timeout_does_not_commit_config(self):
        session = await (await self.model()).connect(options=RealtimeConnectOptions(timeout_ms=10))
        self.connection.ack = False
        with self.assertRaises(TimeoutError):
            await session.update(translation_target_language_code="es")
        self.assertIsNone(session.config.translation_target_language_code)
        self.assertTrue(self.connection.closed)

    async def test_finish_rejects_new_audio_while_waiting_for_existing_update(self):
        session = await (await self.model()).connect()
        self.connection.ack = False
        update = asyncio.create_task(session.update(translation_target_language_code="es"))
        while len(self.connection.sent) < 2:
            await asyncio.sleep(0)
        closing = asyncio.create_task(session.finish())
        await asyncio.sleep(0)
        sending = asyncio.create_task(session.send_audio(AudioFrame(b"\0\0", "audio/pcm")))
        await self.connection.queue.put({"type": "session.updated", "session": {}})
        await asyncio.gather(update, closing)
        with self.assertRaises(ValidationError):
            await sending
        self.assertFalse(any(e["type"] == "input_audio_buffer.append" for e in self.connection.sent))

    async def test_concurrent_finish_is_idempotent(self):
        session = await (await self.model()).connect()
        await asyncio.gather(session.finish(), session.finish(), session.aclose())
        self.assertEqual(sum(e["type"] == "session.finish" for e in self.connection.sent), 1)

    async def test_cancel_finish_releases_connection(self):
        session = await (await self.model(Connection(finish=False))).connect()
        task = asyncio.create_task(session.finish())
        while not any(e["type"] == "session.finish" for e in self.connection.sent):
            await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.connection.closed)

    async def test_provider_failures_and_malformed_audio_are_terminal(self):
        for event in (
            {"type": "error", "error": {"message": "private payload"}},
            {"type": "conversation.item.input_audio_transcription.failed"},
            {"type": "response.done", "response": {"status": "failed"}},
            {"type": "response.done", "response": {"status": "incomplete"}},
            {"type": "response.audio.delta", "delta": "!!!"},
            {"type": "response.audio.delta", "delta": "AA=="},
            {"type": "unknown", "data": "x" * 1048576},
            None,
        ):
            session = await (await self.model()).connect()
            await self.connection.queue.put(event)
            events = [e async for e in session.event_stream()]
            self.assertEqual(events[-1].reason, "error")
            self.assertEqual(events[-2].type, "realtime-error")
            with self.assertRaises(Exception):
                await session.finish()
            self.assertTrue(self.connection.closed)

    async def test_native_events_preserve_unknown_speaker_events(self):
        session = await (await self.model()).connect()
        self.connection.final_events = [{"type": "future.speaker.event", "speaker": {"id": "a", "voice": "b"}}]
        await session.finish()
        self.assertIn(self.connection.final_events[0], [e async for e in session.native_event_stream()])

    async def test_image_and_clear_audio(self):
        session = await (await self.model()).connect()
        jpeg = b"\xff\xd8\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xd9"
        with self.assertRaisesRegex(ValidationError, "before"):
            await session.send_image(jpeg)
        await session.send_audio(AudioFrame(data="AAA=", media_type="audio/pcm"))
        await session.send_image(jpeg)
        self.assertEqual(base64.b64decode(self.connection.sent[-1]["image"]), jpeg)
        with self.assertRaisesRegex(ValidationError, "two images"):
            await session.send_image(jpeg)
        for image in (b"png", b"\xff\xd8\xff" + b"x" * 500000):
            with self.assertRaises(ValidationError):
                await session.send_image(image)
        await session.clear_audio()
        self.assertEqual(self.connection.sent[-1]["type"], "input_audio_buffer.clear")
        await session.finish()

    async def test_reject_audio_format_and_agent_inputs(self):
        session = await (await self.model()).connect()
        for frame in (AudioFrame(b"\0", "audio/pcm"), AudioFrame(b"\0\0", "audio/wav"), AudioFrame(b"\0\0", "audio/pcm", sample_rate_hz=24000), AudioFrame(b"\0\0", "audio/pcm", channels=2), AudioFrame("!", "audio/pcm"), AudioFrame(b"RIFFabcd", "audio/pcm")):
            with self.assertRaises(ValidationError):
                await session.send_audio(frame)
        with self.assertRaises(UnsupportedFeatureError):
            await session.send_text("hello")
        with self.assertRaises(UnsupportedFeatureError):
            await session.send_tool_result(None)
        await session.finish()

    async def test_validation_happens_before_network(self):
        model = await self.model()
        with self.assertRaises(ValidationError):
            await model.connect(RealtimeSessionConfig(translation_target_language_code="invalid"))
        with self.assertRaises(UnsupportedFeatureError):
            await model.connect(options=RealtimeConnectOptions(browser_client=True))
        with self.assertRaises(UnsupportedFeatureError):
            await model.create_browser_token()
        self.assertEqual(self.calls, [])


    async def test_partial_update_preserves_modalities_and_replaces_glossary(self):
        session = await (await self.model()).connect(RealtimeSessionConfig(provider_options={
            "output_modalities": ["text"], "translation": {"corpus": {"phrases": {"old": "previous"}}},
        }))
        await session.update(provider_options={"translation": {"corpus": {"phrases": {"new": "current"}}}})
        body = self.connection.sent[-1]["session"]
        self.assertEqual(body["output_modalities"], ["text"])
        self.assertEqual(body["translation"]["corpus"]["phrases"], {"new": "current"})
        await session.finish()

    async def test_cancel_setup_releases_connection(self):
        model = await self.model(Connection(ack=False))
        task = asyncio.create_task(model.connect())
        while not self.connection.sent:
            await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.connection.closed)

    async def test_send_timeout_closes_transport(self):
        session = await (await self.model()).connect(options=RealtimeConnectOptions(timeout_ms=10))

        async def blocked(_):
            await asyncio.Event().wait()

        self.connection.send_json = blocked
        with self.assertRaises(TimeoutError):
            await session.send_audio(AudioFrame(b"\0\0", "audio/pcm"))
        self.assertTrue(self.connection.closed)
        with self.assertRaises(TimeoutError):
            await session.finish()

    async def test_update_provider_error_does_not_commit_configuration(self):
        session = await (await self.model()).connect()

        async def rejected(_):
            await self.connection.queue.put({"type": "error", "error": {"code": "invalid_value"}})

        self.connection.send_json = rejected
        with self.assertRaises(ConnectionError):
            await session.update(translation_target_language_code="es")
        self.assertIsNone(session.config.translation_target_language_code)
        self.assertTrue(self.connection.closed)

    async def test_finish_waits_for_pending_update(self):
        session = await (await self.model()).connect()
        self.connection.ack = False
        update = asyncio.create_task(session.update(translation_target_language_code="es"))
        while len(self.connection.sent) < 2:
            await asyncio.sleep(0)
        closing = asyncio.create_task(session.finish())
        await asyncio.sleep(0)
        self.assertFalse(any(e["type"] == "session.finish" for e in self.connection.sent))
        await self.connection.queue.put({"type": "session.updated", "session": {}})
        await asyncio.gather(update, closing)
        self.assertTrue(self.connection.closed)

    async def test_native_stream_overrun_is_explicit(self):
        session = await (await self.model()).connect()
        self.connection.final_events = [{"type": "unknown", "index": i} for i in range(1001)]
        await session.finish()
        with self.assertRaisesRegex(ValidationError, "history"):
            _ = [e async for e in session.native_event_stream()]

    async def test_oversized_and_truncated_jpeg_rejected(self):
        session = await (await self.model()).connect()
        await session.send_audio(AudioFrame(b"\0\0", "audio/pcm"))
        for jpeg in (b"\xff\xd8\xff\xc0\x00\x0b\x08\x08\x00\x08\x00\x01\x01\x11\x00\xff\xd9", b"\xff\xd8\xff\xd9", b"\xff\xd8\xff\xe0\xff\xffbroken\xff\xd9"):
            with self.assertRaises(ValidationError):
                await session.send_image(jpeg)
        await session.finish()


class QwenRealtimeConfigurationTests(TestCase):
    def test_realtime_model_cannot_be_used_as_a_language_model(self):
        with self.assertRaises(UnsupportedFeatureError):
            create_qwen(api_key="test")(MODEL_ID)

    def test_only_exact_model_supported(self):
        for model in ("qwen3.5-livetranslate-flash-realtime", "qwen3.8-omni-flash", MODEL_ID + "-fake"):
            with self.assertRaises(UnsupportedFeatureError):
                create_qwen(api_key="test").realtime_model(model)

    def test_endpoints(self):
        self.assertIn("dashscope-intl.aliyuncs.com/api-ws", realtime_url("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "intl", None, None))
        self.assertIn("test.cn-beijing.maas", realtime_url("", "cn", "test", None))
        self.assertIn("example.com/custom?tenant=1&model=", realtime_url("", "intl", None, "wss://example.com/custom?tenant=1"))
        for url in ("ws://example.com", "wss://user:password@example.com", "wss://example.com/#x", "wss://example.com?model=wrong", "wss://example.com:bad", "wss://example.com\n"):
            with self.assertRaises(ConfigurationError):
                realtime_url("", "intl", None, url)
        with self.assertRaises(ConfigurationError):
            realtime_url("", "us", "test", None)
        with self.assertRaises(ConfigurationError):
            realtime_url("", "cn", "abc/path", None)

    def test_settings_and_voice_clone(self):
        payload = session_payload(RealtimeSessionConfig(voice="default", input_sample_rate_hz=16000, output_sample_rate_hz=24000, turn_detection={"type": "speaker_detection"}, provider_options={"enable_voice_clone": True, "voice_clone_options": {"frequency": "always"}}))
        self.assertEqual(payload["audio"]["output"]["voice"], "default")
        self.assertEqual(payload["audio"]["input"]["format"], {"sample_rate": 16000})
        self.assertEqual(payload["voice_clone_options"]["frequency"], "always")

    def test_invalid_options_and_conflicts(self):
        invalid = [
            {"output_modalities": ["audio"]}, {"modalities": ["text"]},
            {"translation": {"same_language_skip_options": {}}},
            {"translation": {"language": "xx"}}, {"translation": {"language": "yue"}},
            {"translation": {"corpus": {"phrases": {"x": 1}}}},
            {"input_audio_transcription": None}, {"audio": {"input": {"format": {"type": "opus"}}}},
            {"audio": {"input": {"turn_detection": None}}},
            {"audio": {"input": {"turn_detection": {"type": "server_vad"}}}},
            {"audio": {"input": {"turn_detection": {"threshold": float("nan")}}}},
            {"audio": {"input": {"turn_detection": {"silence_duration_ms": 0}}}},
            {"enable_voice_clone": "yes"}, {"voice_clone_options": {"frequency": "always"}},
            {"enable_voice_clone": True, "audio": {"output": {"voice": "Tina"}}},
            {"enable_voice_clone": True, "voice_clone_options": {"frequency": "never"}},
        ]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises((ValidationError, UnsupportedFeatureError)):
                session_payload(RealtimeSessionConfig(provider_options=options))
        with self.assertRaisesRegex(ValidationError, "Conflicting"):
            session_payload(RealtimeSessionConfig(translation_target_language_code="es", provider_options={"translation": {"language": "en"}}))
