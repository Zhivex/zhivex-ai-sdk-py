"""Wire regressions for official September 2026 provider contracts (offline)."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from unittest import IsolatedAsyncioTestCase, TestCase

from zhivex_ai import (
    ImagePart,
    ModelMessage,
    ReasoningConfig,
    TextPart,
    RealtimeSessionConfig,
    UnsupportedFeatureError,
    ValidationError,
    ProviderHTTPError,
    ParseError,
    create_anthropic,
    create_azure_openai,
    create_deepseek,
    create_gemini,
    create_meta,
    create_openai,
    generate_text,
    stream_text,
    generate_object,
    tool,
)
from zhivex_ai.catalog import default_model_catalog
from tests.test_anthropic_provider import FakeResponse
from tests.test_realtime import FakeRealtimeConnection


class SeptemberProviderUpdatesTests(IsolatedAsyncioTestCase):
    async def test_flash_vision_aliases_preserve_wire_identity_and_reasoning(self):
        requests = []

        async def fetch(url, **kwargs):
            requests.append(kwargs["json_body"])
            return FakeResponse(
                200,
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            )

        provider = create_deepseek(api_key="test", fetch=fetch)
        messages = [
            ModelMessage(
                role="user",
                parts=[
                    TextPart(text="Describe"),
                    ImagePart(image="YWJj", media_type="image/png"),
                ],
            )
        ]
        for model in (
            "deepseek-flash",
            "deepseek-v4-flash",
            "deepseek-v4-flash-vision-exp",
        ):
            for effort, wire in [
                ("minimal", "low"),
                ("low", "low"),
                ("medium", "high"),
                ("xhigh", "high"),
                ("max", "max"),
                ("ultra", "max"),
            ]:
                await generate_text(
                    model=provider(model),
                    messages=messages,
                    reasoning=ReasoningConfig(effort=effort),
                )
                self.assertEqual(requests[-1]["model"], model)
                self.assertEqual(requests[-1]["reasoning_effort"], wire)
                self.assertEqual(
                    requests[-1]["messages"][0]["content"][1]["image_url"]["url"],
                    "data:image/png;base64,YWJj",
                )
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(model=provider("deepseek-v4-pro"), messages=messages)
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider("deepseek-flash"),
                messages=[ModelMessage(role="assistant", parts=messages[0].parts)],
            )

    async def test_flash_top_p_is_effective_or_rejected_not_silently_ignored(self):
        requests = []

        async def fetch(url, **kwargs):
            requests.append(kwargs["json_body"])
            return FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})

        model = create_deepseek(api_key="test", fetch=fetch).native.language_model(
            "deepseek-flash"
        )
        await generate_text(
            model=model, prompt="test", provider_options={"top_p": 0.98}
        )
        self.assertEqual(requests[-1]["top_p"], 0.98)
        self.assertNotEqual(requests[-1].get("thinking"), {"type": "disabled"})
        for value in (0.9, True, float("nan"), "0.98"):
            with self.assertRaises(ValidationError):
                await generate_text(
                    model=model, prompt="test", provider_options={"top_p": value}
                )
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=model,
                prompt="test",
                reasoning=ReasoningConfig(effort="none"),
                provider_options={"top_p": 0.98},
            )
        self.assertEqual(len(requests), 1)

    async def test_spark_13_chat_and_stream_use_exact_id(self):
        requests = []

        async def fetch(url, **kwargs):
            requests.append((url, kwargs["json_body"]))
            if kwargs.get("stream"):
                return FakeResponse(
                    200,
                    body_text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
                )
            return FakeResponse(
                200,
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            )

        model = create_meta(api_key="test", fetch=fetch)("muse-spark-1.3")
        await generate_text(
            model=model, prompt="test", reasoning=ReasoningConfig(effort="max")
        )
        self.assertEqual(requests[-1][1]["reasoning_effort"], "max")
        async with stream_text(model=model, prompt="test") as result:
            self.assertEqual("".join([x async for x in result.text_stream()]), "ok")
        self.assertTrue(all(body["model"] == "muse-spark-1.3" for _, body in requests))
        self.assertTrue(all(url.endswith("/chat/completions") for url, _ in requests))

    async def test_spark13_structured_output_and_tool_replay(self):
        requests = []

        async def fetch(url, **kwargs):
            body = kwargs["json_body"]
            requests.append(body)
            if body.get("response_format"):
                payload = {"content": '{"ok": true}'}
            elif len(requests) == 2:
                payload = {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "check", "arguments": "{}"},
                        }
                    ],
                }
            else:
                payload = {"content": "done"}
            return FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": payload,
                            "finish_reason": "tool_calls"
                            if payload.get("tool_calls")
                            else "stop",
                        }
                    ]
                },
            )

        provider = create_meta(api_key="test", fetch=fetch)
        model = provider("muse-spark-1.3")
        await generate_object(
            model=model,
            prompt="test",
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        )
        self.assertEqual(requests[-1]["response_format"]["type"], "json_schema")
        result = await generate_text(
            model=model,
            prompt="check",
            tools={
                "check": tool(
                    name="check",
                    schema={"type": "object", "properties": {}},
                    execute=lambda value: {"ok": True},
                )
            },
            max_steps=2,
        )
        self.assertEqual(result.text, "done")
        self.assertTrue(
            any(m.get("tool_call_id") == "call_1" for m in requests[-1]["messages"])
        )
        self.assertTrue(all(r["model"] == "muse-spark-1.3" for r in requests))

    async def test_openai_cache_diagnostics_native_round_trip(self):
        requests = []
        diagnostic = {"type": "cache_miss", "reason": "tools_changed"}

        async def fetch(url, **kwargs):
            requests.append(kwargs["json_body"])
            return FakeResponse(
                200,
                {
                    "id": "resp_new",
                    "status": "completed",
                    "output": [],
                    "prompt_cache_diagnostics": diagnostic,
                },
            )

        result = (
            await create_openai(api_key="test", fetch=fetch)
            .native.responses()
            .create(
                {
                    "model": "gpt-6-astra",
                    "input": "test",
                    "prompt_cache_options": {"comparison_response_id": "resp_old"},
                }
            )
        )
        self.assertEqual(
            requests[-1]["prompt_cache_options"], {"comparison_response_id": "resp_old"}
        )
        self.assertEqual(result["prompt_cache_diagnostics"], diagnostic)
        self.assertNotIn("previous_response_id", requests[-1])

    async def test_gemini_extended_preserves_simultaneous_content_and_tool_events(self):
        from zhivex_ai.providers.gemini import _gemini_realtime_parse_event
        from zhivex_ai import (
            RealtimeAudioOutputEvent,
            RealtimeToolCallEvent,
            RealtimeResponseCompletedEvent,
        )

        payload = {
            "serverContent": {
                "modelTurn": {
                    "parts": [{"inlineData": {"mimeType": "audio/pcm", "data": "YWJj"}}]
                },
                "turnComplete": True,
                "interaction_status": "IN_PROGRESS",
            },
            "toolCall": {
                "functionCalls": [{"id": "call1", "name": "check", "args": {}}]
            },
        }
        events = _gemini_realtime_parse_event(payload)
        self.assertTrue(any(isinstance(e, RealtimeAudioOutputEvent) for e in events))
        self.assertTrue(any(isinstance(e, RealtimeToolCallEvent) for e in events))
        self.assertFalse(
            any(isinstance(e, RealtimeResponseCompletedEvent) for e in events)
        )
        events = _gemini_realtime_parse_event(
            {"serverContent": {"interaction_status": "IDLE"}}
        )
        self.assertTrue(
            any(isinstance(e, RealtimeResponseCompletedEvent) for e in events)
        )

    async def test_image25_generate_and_edit_forward_quality(self):
        requests = []

        async def fetch(url, **kwargs):
            requests.append((url, kwargs))
            return FakeResponse(200, {"data": [{"b64_json": "YWJj"}]})

        client = create_openai(api_key="test", fetch=fetch).native.images()
        for model in ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare"):
            result = await client.generate(prompt="test", model=model, quality="max")
            self.assertEqual(len(result.images), 1)
            self.assertEqual(requests[-1][1]["json_body"]["quality"], "max")
            self.assertEqual(requests[-1][1]["json_body"]["model"], model)
            await client.edit(
                prompt="test",
                model=model,
                quality="xhigh",
                image=b"abc",
                image_filenames="a.png",
            )
            self.assertTrue(requests[-1][0].endswith("/images/edits"))
            self.assertEqual(requests[-1][1]["body"]["data"]["quality"], "xhigh")

    async def test_lyria35_music_preserves_audio(self):
        requests = []

        async def fetch(url, **kwargs):
            requests.append((url, kwargs["json_body"]))
            return FakeResponse(
                200,
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "audio/wav",
                                            "data": "YWJj",
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            )

        result = (
            await create_gemini(api_key="test", fetch=fetch)
            .native.media()
            .generate_music(prompt="test", model="lyria-3.5")
        )
        self.assertIn("/models/lyria-3.5:generateContent", requests[-1][0])
        self.assertEqual(result.media[0].media_type, "audio/wav")

    async def test_anthropic_signed_compaction_replay_and_stream(self):
        block = {
            "type": "compaction",
            "content": "summary",
            "signature": "opaque-signature",
        }
        requests = []

        async def fetch(url, **kwargs):
            requests.append(deepcopy(kwargs))
            if kwargs.get("stream"):
                events = [
                    ("content_block_start", {"index": 0, "content_block": block}),
                    ("content_block_stop", {"index": 0}),
                    ("message_delta", {"delta": {"stop_reason": "compaction"}}),
                    ("message_stop", {}),
                ]
                return FakeResponse(
                    200,
                    body_text="".join(
                        "event: " + name + "\ndata: " + json.dumps(data) + "\n\n"
                        for name, data in events
                    ),
                )
            return FakeResponse(200, {"content": [block], "stop_reason": "compaction"})

        provider = create_anthropic(api_key="test", fetch=fetch)
        model = provider.native.language_model("claude-fable-5-1")
        result = await generate_text(
            model=model,
            prompt="summarize",
            provider_options={"compaction": {"type": "summarize"}},
        )
        self.assertEqual(result.messages[-1].parts[0].data["block"], block)
        await generate_text(
            model=model,
            messages=[
                result.messages[-1],
                ModelMessage(role="user", parts=[TextPart(text="continue")]),
            ],
        )
        self.assertEqual(requests[-1]["json_body"]["messages"][0]["content"], [block])
        self.assertIn("compact-2026-09-04", requests[-1]["headers"]["anthropic-beta"])
        async with stream_text(
            model=model,
            prompt="summarize",
            provider_options={"compaction": {"type": "summarize"}},
        ) as stream:
            events = [e async for e in stream.event_stream()]
        self.assertTrue(
            any(
                getattr(e, "data", {}).get("block") == block
                for e in events
                if isinstance(getattr(e, "data", None), dict)
            )
        )
        raw = provider.native.messages()
        body = {
            "model": "claude-fable-5-1",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "test"}],
        }
        original = deepcopy(body)
        self.assertEqual((await raw.compact(body))["content"][0], block)
        self.assertEqual(body, original)
        await raw.compact({**body, "compaction": None})
        self.assertEqual(requests[-1]["json_body"]["compaction"], {"type": "summarize"})
        for invalid in (
            {"context_management": {}},
            {"output_config": "invalid"},
            {"tool_choice": "invalid"},
            {"stop_sequences": []},
            {"tool_choice": {"type": "any"}},
            {"output_config": {"format": {"type": "json_schema"}}},
        ):
            with self.assertRaises(ValidationError):
                await raw.compact({**body, **invalid})

    async def test_gemini38_live_setup_and_invalid_thinking_before_connect(self):
        connections = []

        async def factory(url, headers, options):
            connection = FakeRealtimeConnection([{"setupComplete": {}}])
            connections.append(connection)
            return connection

        p = create_gemini(api_key="test", realtime_connection_factory=factory)
        for name in ("gemini-3.8-live", "gemini-3.8-live-extended-thinking"):
            session = await p.native.realtime_model(name).connect()
            self.assertEqual(
                connections[-1].sent[0]["setup"]["generationConfig"][
                    "responseModalities"
                ],
                ["AUDIO"],
            )
            await session.send_text("continue")
            self.assertTrue(connections[-1].sent[-1]["clientContent"]["turnComplete"])
            await session.aclose()
        for name, level in [
            ("gemini-3.8-live", "low"),
            ("gemini-3.8-live-extended-thinking", "minimal"),
        ]:
            with self.assertRaises(UnsupportedFeatureError):
                await p.native.realtime_model(name).connect(
                    RealtimeSessionConfig(
                        provider_options={
                            "generationConfig": {
                                "thinkingConfig": {"thinkingLevel": level}
                            }
                        }
                    )
                )
        self.assertEqual(len(connections), 2)

    async def test_native_agents_http_identity_events_and_no_automatic_retry(self):
        requests = []

        async def fetch(url, **kwargs):
            requests.append((url, deepcopy(kwargs)))
            if kwargs.get("stream"):
                return FakeResponse(
                    200,
                    body_text='data: {"type":"agent.session.completed"}\n\ndata: [DONE]\n\n',
                )
            return FakeResponse(200, {"id": "sess_test", "data": []})

        p = create_openai(api_key="test", fetch=fetch)
        client = p.native.agent_sessions()
        self.assertIs(client, p.native.agent_sessions())
        body = {
            "agent": {"model": "gpt-6-astra"},
            "environment": {"type": "openai_hosted"},
        }
        await client.create(body)
        self.assertEqual(requests[-1][0], "https://api.openai.com/v1/agents/sessions")
        self.assertEqual(requests[-1][1]["headers"]["OpenAI-Beta"], "agents=v1")
        await client.retrieve("id/?x")
        self.assertTrue(requests[-1][0].endswith("/id%2F%3Fx"))
        await client.send_events(
            "sess_test", [{"type": "agent.session.input.message", "input": "test"}]
        )
        await client.cancel("sess_test")
        self.assertEqual(
            requests[-1][1]["json_body"]["events"],
            [{"type": "agent.session.input.cancel"}],
        )
        self.assertEqual(
            [e async for e in client.events("sess_test")],
            [{"type": "agent.session.completed"}],
        )
        await client.items("sess_test", after="item & 1")
        self.assertIn("after=item+%26+1", requests[-1][0])
        await client.delete("sess_test")
        self.assertEqual(requests[-1][1]["method"], "DELETE")
        self.assertFalse(
            create_azure_openai(
                api_key="test", endpoint="https://example.test"
            ).native_support.agent_sessions
        )
        calls = []

        async def failing(url, **kwargs):
            calls.append(url)
            return FakeResponse(503, {"error": "unavailable"})

        with self.assertRaises(ProviderHTTPError):
            await (
                create_openai(api_key="test", fetch=failing)
                .native.agent_sessions()
                .create(body)
            )
        self.assertEqual(len(calls), 1)

    async def test_live_http_and_agents_void_responses(self):
        from zhivex_ai._http import BufferedResponse

        requests = []

        async def fetch(url, **kwargs):
            requests.append((url, kwargs))
            if url.endswith(("/hangup", "/events")):
                return BufferedResponse(200, b"")
            if url.endswith("/content"):
                return BufferedResponse(200, b"RIFF-audio")
            return BufferedResponse(200, b'{"id":"s","transport":{"sdp":"answer"}}')

        provider = create_openai(api_key="test", fetch=fetch)
        live = provider.native.live()
        session = {"model": "gpt-live-1"}
        transport = {"type": "webrtc", "sdp": "offer"}
        result = await live.create(session=session, transport=transport)
        self.assertEqual(result["transport"]["sdp"], "answer")
        self.assertEqual(
            requests[-1][1]["json_body"], {"session": session, "transport": transport}
        )
        await live.fork("s", {"transport": transport})
        self.assertTrue(requests[-1][0].endswith("/live/sessions/s/fork"))
        self.assertIsNone(await live.hangup("s"))
        self.assertEqual(await live.download_recording("s"), b"RIFF-audio")
        self.assertEqual(requests[-1][1]["method"], "GET")
        self.assertIsNone(await provider.native.agent_sessions().send_events("s", []))
        self.assertIsNone(await provider.native.agent_sessions().cancel("s"))

    async def test_agents_subscription_closes_when_consumer_stops(self):
        from contextlib import aclosing

        class ClosableResponse(FakeResponse):
            closed = False

            async def aclose(self):
                self.closed = True

        response = ClosableResponse(
            200, body_text='data: {"type":"first"}\n\ndata: {"type":"second"}\n\n'
        )

        async def fetch(url, **kwargs):
            self.assertEqual(kwargs["headers"]["Accept"], "text/event-stream")
            return response

        client = create_openai(api_key="test", fetch=fetch).native.agent_sessions()
        async with aclosing(client.events("s")) as events:
            async for event in events:
                self.assertEqual(event["type"], "first")
                break
        self.assertTrue(response.closed)

    async def test_live_uses_new_protocol_and_confirms_final_usage(self):
        connections = []

        async def factory(url, headers, options):
            self.assertEqual(url, "wss://api.openai.com/v1/live/sessions")
            self.assertNotIn("OpenAI-Beta", headers)
            c = FakeRealtimeConnection(
                [
                    {"type": "session.started", "session": {"id": "s"}},
                    {"type": "session.closed", "usage": {"seconds": 4}},
                ]
            )
            connections.append(c)
            return c

        p = create_openai(api_key="test", realtime_connection_factory=factory)
        session = await p.native.live().connect(
            {"model": "gpt-live-1", "delegation": {"type": "client"}}
        )
        self.assertEqual(connections[0].sent[0]["type"], "session.start")
        await session.send({"type": "session.input_audio.append", "audio": "YWJj"})
        final = await session.finish()
        self.assertEqual(final["usage"]["seconds"], 4)
        self.assertTrue(connections[0].closed)
        self.assertEqual(connections[0].sent[-1], {"type": "session.close"})
        with self.assertRaises(ValidationError):
            await session.send({"type": "session.input_audio.append"})

    async def test_live_startup_and_finalization_failures_close_socket(self):
        connections = []

        async def factory(url, headers, options):
            c = FakeRealtimeConnection(
                [{"type": "error"}]
                if not connections
                else [{"type": "session.started"}]
            )
            connections.append(c)
            return c

        client = create_openai(
            api_key="test", realtime_connection_factory=factory
        ).native.live()
        with self.assertRaises(ParseError):
            await client.connect({"model": "gpt-live-1"})
        self.assertTrue(connections[-1].closed)
        session = await client.connect({"model": "gpt-live-1"})
        with self.assertRaises(ParseError):
            await session.finish()
        self.assertTrue(connections[-1].closed)
        self.assertIsNone(session.final_event)

    async def test_live_finalization_timeout_releases_transport(self):
        class HangingConnection(FakeRealtimeConnection):
            async def recv_json(self):
                if self._incoming:
                    return self._incoming.pop(0)
                await asyncio.Event().wait()

        c = HangingConnection([{"type": "session.started"}])

        async def factory(url, headers, options):
            return c

        session = (
            await create_openai(api_key="test", realtime_connection_factory=factory)
            .native.live()
            .connect({"model": "gpt-live-1"})
        )
        with self.assertRaises(TimeoutError):
            await session.finish(timeout_ms=1)
        self.assertTrue(c.closed)


class SeptemberCatalogTests(TestCase):
    def test_snapshot_keeps_provider_and_surface_boundaries(self):
        cases = [
            ("deepseek", "deepseek-flash", "language"),
            ("meta", "muse-spark-1.3", "language"),
            ("openai", "gpt-image-2.5-sunburst", "image"),
            ("openai", "gpt-image-2.5-flare", "image"),
            ("gemini", "gemini-3.8-live", "realtime"),
            ("gemini", "gemini-3.8-live-extended-thinking", "realtime"),
            ("gemini", "lyria-3.5", "media"),
        ]
        for provider, model, surface in cases:
            entry = default_model_catalog.find(provider, model)
            self.assertEqual(entry.api_surface, surface)
            self.assertEqual(entry.verified_at, "2026-09-16")
            self.assertEqual(entry.support_evidence, "offline-contract")
        for model in ("deepseek-v4-flash", "deepseek-v4-flash-vision-exp"):
            entry = default_model_catalog.find("deepseek", model)
            self.assertEqual(entry.availability, "deprecated")
            self.assertEqual(entry.replacement_model_id, "deepseek-flash")
            self.assertTrue(entry.capabilities.vision)
        self.assertIsNone(default_model_catalog.find("vertex", "gemini-3.8-live"))
