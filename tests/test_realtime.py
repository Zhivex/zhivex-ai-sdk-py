from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    Agent,
    AgentCancellationToken,
    AgentRunCancelled,
    AgentTextDeltaEvent,
    ApprovalDecision,
    AudioFrame,
    GenerateResult,
    ModelCapabilities,
    ModelGenerateInput,
    RealtimeAudioOutputEvent,
    RealtimeConnectOptions,
    RealtimeResponseCompletedEvent,
    RealtimeSessionEndedEvent,
    RealtimeSessionConfig,
    RealtimeTextDeltaEvent,
    RealtimeToolCallEvent,
    RealtimeToolResultEvent,
    RealtimeTranscriptEvent,
    ToolExecutionResult,
    ToolDefinition,
    ToolExecutionOptions,
    ToolExecutionOutcomeUnknown,
    ToolApprovalRequest,
    ToolCall,
    TextPart,
    UnsupportedFeatureError,
    ValidationError,
    cancel_agent_run,
    create_gemini,
    create_openai,
    create_in_memory_agent_memory_store,
    create_in_memory_agent_run_store,
    create_text_message,
    resume_agent_run,
    stream_live_agent,
)


class FakeRealtimeConnection:
    def __init__(self, incoming: list[dict[str, Any]]) -> None:
        self._incoming = list(incoming)
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def recv_json(self) -> Any:
        if self.closed:
            return None
        if not self._incoming:
            return None
        return self._incoming.pop(0)

    async def close(self) -> None:
        self.closed = True


class RealtimeProviderTests(IsolatedAsyncioTestCase):
    async def test_realtime_session_rejects_oversized_provider_message(self) -> None:
        async def connection_factory(url: str, headers: dict[str, str], options: RealtimeConnectOptions | None):
            return FakeRealtimeConnection([{"server_content": {"model_turn": {"parts": [{"text": "x" * (1024 * 1024 + 1)}]}}}])

        provider = create_gemini(api_key="test", realtime_connection_factory=connection_factory)
        session = await provider.realtime_model("gemini-3.1-flash-live-preview").connect()
        events = []
        async for event in session.event_stream():
            events.append(event)
            if isinstance(event, RealtimeSessionEndedEvent):
                break
        await session.aclose()

        ended = [event for event in events if isinstance(event, RealtimeSessionEndedEvent)]
        self.assertTrue(ended)
        self.assertEqual(ended[0].reason, "error")

    async def test_openai_realtime_connects_sends_payloads_and_creates_browser_token(self) -> None:
        requests: list[dict[str, Any]] = []
        connections: list[FakeRealtimeConnection] = []
        connection_meta: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any] | None,
            timeout_ms: int | None,
            method: str = "POST",
            body: Any = None,
            stream: bool = False,
        ):
            requests.append({"url": url, "headers": headers, "json": json_body})

            class FakeResponse:
                status_code = 200
                headers = {"content-type": "application/json"}

                async def json(self) -> Any:
                    return {"client_secret": {"value": "token-123", "expires_at_ms": 999}}

                async def text(self) -> str:
                    return json.dumps(await self.json())

                async def read(self) -> bytes:
                    return json.dumps(await self.json()).encode()

                async def iter_lines(self):
                    if False:
                        yield ""

            return FakeResponse()

        async def connection_factory(url: str, headers: dict[str, str], options: RealtimeConnectOptions | None):
            connection_meta.append({"url": url, "headers": headers, "options": options})
            connection = FakeRealtimeConnection(
                [
                    {"type": "conversation.item.input_audio_transcription.completed", "transcript": "hola"},
                    {"type": "response.text.delta", "delta": "mundo"},
                    {"type": "response.output_audio.delta", "delta": "AQI="},
                    {"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{\"q\":\"status\"}"}},
                    {"type": "response.done"},
                ]
            )
            connections.append(connection)
            return connection

        provider = create_openai(
            api_key="test",
            fetch=fetch,
            realtime_connection_factory=connection_factory,
        )
        model = provider.realtime_model("gpt-realtime-2.1")
        session = await model.connect(
            RealtimeSessionConfig(voice="alloy", instructions="Speak briefly."),
            RealtimeConnectOptions(timeout_ms=500),
        )

        await session.send_audio(AudioFrame(data=b"\x00\x01", media_type="audio/pcm", is_final=True))
        await session.send_text("hi")
        events = []
        async for event in session.event_stream():
            events.append(event)
            if isinstance(event, RealtimeResponseCompletedEvent):
                break
        await session.aclose()

        self.assertIn("/realtime?model=gpt-realtime-2.1", connection_meta[0]["url"])
        self.assertNotIn("OpenAI-Beta", connection_meta[0]["headers"])
        self.assertEqual(connections[0].sent[0]["type"], "session.update")
        self.assertEqual(connections[0].sent[1]["type"], "input_audio_buffer.append")
        self.assertEqual(connections[0].sent[2]["type"], "input_audio_buffer.commit")
        self.assertEqual(connections[0].sent[3]["type"], "response.create")
        self.assertTrue(any(isinstance(event, RealtimeTranscriptEvent) and event.text == "hola" for event in events))
        self.assertTrue(any(isinstance(event, RealtimeTextDeltaEvent) and event.text_delta == "mundo" for event in events))
        self.assertTrue(any(isinstance(event, RealtimeAudioOutputEvent) and event.audio == b"\x01\x02" for event in events))
        self.assertTrue(any(isinstance(event, RealtimeToolCallEvent) and event.tool_call.name == "lookup" for event in events))
        self.assertTrue(any(isinstance(event, RealtimeResponseCompletedEvent) for event in events))

        token = await model.create_browser_token(RealtimeSessionConfig(voice="alloy"))
        self.assertEqual(token.value, "token-123")
        self.assertEqual(requests[0]["url"], "https://api.openai.com/v1/realtime/client_secrets")
        self.assertEqual(requests[0]["json"]["session"]["type"], "realtime")
        self.assertEqual(requests[0]["json"]["session"]["model"], "gpt-realtime-2.1")
        self.assertEqual(requests[0]["json"]["session"]["audio"]["output"]["voice"], "alloy")

    async def test_gemini_realtime_normalizes_server_content(self) -> None:
        async def connection_factory(url: str, headers: dict[str, str], options: RealtimeConnectOptions | None):
            return FakeRealtimeConnection(
                [
                    {
                        "server_content": {
                            "model_turn": {
                                "parts": [
                                    {"text": "respuesta"},
                                    {"functionCall": {"name": "weather", "args": {"city": "BA"}}},
                                ]
                            },
                            "output_transcription": {"text": "respuesta final"},
                            "turn_complete": True,
                        }
                    }
                ]
            )

        provider = create_gemini(api_key="test", realtime_connection_factory=connection_factory)
        session = await provider.realtime_model("gemini-3.1-flash-live-preview").connect(
            RealtimeSessionConfig(output_audio_media_type="audio/pcm")
        )
        events = []
        async for event in session.event_stream():
            events.append(event)
            if isinstance(event, RealtimeResponseCompletedEvent):
                break
        await session.aclose()

        self.assertTrue(any(isinstance(event, RealtimeTextDeltaEvent) and event.text_delta == "respuesta" for event in events))
        self.assertTrue(any(isinstance(event, RealtimeToolCallEvent) and event.tool_call.name == "weather" for event in events))
        self.assertTrue(any(isinstance(event, RealtimeTranscriptEvent) and event.text == "respuesta final" for event in events))
        self.assertTrue(any(isinstance(event, RealtimeResponseCompletedEvent) for event in events))

    async def test_gemini_realtime_accepts_ephemeral_access_token(self) -> None:
        seen: list[dict[str, Any]] = []

        async def connection_factory(url: str, headers: dict[str, str], options: RealtimeConnectOptions | None):
            seen.append({"url": url, "headers": headers})
            return FakeRealtimeConnection([])

        provider = create_gemini(api_key="server-key", realtime_connection_factory=connection_factory)
        session = await provider.realtime_model("gemini-3.1-flash-live-preview").connect(
            RealtimeSessionConfig(provider_options={"access_token": "authTokens/ephemeral"})
        )
        await session.aclose()

        self.assertIn("access_token=authTokens%2Fephemeral", seen[0]["url"])
        self.assertNotIn("key=server-key", seen[0]["url"])

    async def test_gemini_live_translate_setup_and_events_are_supported(self) -> None:
        connections: list[FakeRealtimeConnection] = []

        async def connection_factory(url: str, headers: dict[str, str], options: RealtimeConnectOptions | None):
            connection = FakeRealtimeConnection(
                [
                    {
                        "serverContent": {
                            "modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm", "data": "AQI="}}]},
                            "inputTranscription": {"text": "hello", "languageCode": "en"},
                            "outputTranscription": {"text": "hola", "languageCode": "es"},
                            "turnComplete": True,
                        }
                    }
                ]
            )
            connections.append(connection)
            return connection

        provider = create_gemini(api_key="test", realtime_connection_factory=connection_factory)
        session = await provider.realtime_model("gemini-3.5-live-translate-preview").connect(
            RealtimeSessionConfig(
                input_audio_media_type="audio/pcm;rate=16000",
                output_audio_media_type="audio/pcm",
                translation_target_language_code="es",
                translation_echo_target_language=True,
            )
        )
        setup = connections[0].sent[0]["setup"]
        generation_config = setup["generationConfig"]
        self.assertEqual(generation_config["responseModalities"], ["AUDIO"])
        self.assertEqual(setup["inputAudioTranscription"], {})
        self.assertEqual(setup["outputAudioTranscription"], {})
        self.assertEqual(
            generation_config["translationConfig"],
            {"targetLanguageCode": "es", "echoTargetLanguage": True},
        )

        await session.send_audio(AudioFrame(data=b"\x00\x01", media_type="audio/pcm;rate=16000", is_final=True))
        self.assertEqual(connections[0].sent[1]["realtimeInput"]["audio"]["mimeType"], "audio/pcm;rate=16000")
        self.assertEqual(connections[0].sent[2], {"realtimeInput": {"audioStreamEnd": True}})

        events = []
        async for event in session.event_stream():
            events.append(event)
            if isinstance(event, RealtimeResponseCompletedEvent):
                break
        await session.aclose()

        self.assertTrue(any(isinstance(event, RealtimeAudioOutputEvent) and event.audio == b"\x01\x02" for event in events))
        self.assertTrue(any(isinstance(event, RealtimeTranscriptEvent) and event.role == "user" and event.text == "hello" for event in events))
        self.assertTrue(any(isinstance(event, RealtimeTranscriptEvent) and event.role == "assistant" and event.text == "hola" for event in events))

    async def test_gemini_live_translate_merges_provider_translation_config(self) -> None:
        connections: list[FakeRealtimeConnection] = []

        async def connection_factory(url: str, headers: dict[str, str], options: RealtimeConnectOptions | None):
            connection = FakeRealtimeConnection([])
            connections.append(connection)
            return connection

        provider = create_gemini(api_key="test", realtime_connection_factory=connection_factory)
        session = await provider.realtime_model("gemini-3.5-live-translate-preview").connect(
            RealtimeSessionConfig(
                translation_target_language_code="pl",
                translation_echo_target_language=True,
                provider_options={
                    "generationConfig": {
                        "translationConfig": {"targetLanguageCode": "pl"},
                        "temperature": 0,
                    }
                },
            )
        )
        await session.aclose()

        generation_config = connections[0].sent[0]["setup"]["generationConfig"]
        self.assertEqual(generation_config["temperature"], 0)
        self.assertEqual(
            generation_config["translationConfig"],
            {"targetLanguageCode": "pl", "echoTargetLanguage": True},
        )

    async def test_gemini_live_translate_rejects_conflicting_translation_config(self) -> None:
        async def connection_factory(url: str, headers: dict[str, str], options: RealtimeConnectOptions | None):
            raise AssertionError("connection should not be opened")

        provider = create_gemini(api_key="test", realtime_connection_factory=connection_factory)
        with self.assertRaises(ValidationError):
            await provider.realtime_model("gemini-3.5-live-translate-preview").connect(
                RealtimeSessionConfig(
                    translation_target_language_code="es",
                    provider_options={"generationConfig": {"translationConfig": {"targetLanguageCode": "pl"}}},
                )
            )

    async def test_gemini_live_translate_rejects_agent_features_and_text_input(self) -> None:
        connections: list[FakeRealtimeConnection] = []

        async def connection_factory(url: str, headers: dict[str, str], options: RealtimeConnectOptions | None):
            connection = FakeRealtimeConnection([])
            connections.append(connection)
            return connection

        provider = create_gemini(api_key="test", realtime_connection_factory=connection_factory)
        model = provider.realtime_model("gemini-3.5-live-translate-preview")
        with self.assertRaises(UnsupportedFeatureError):
            await model.connect(RealtimeSessionConfig(instructions="Translate politely."))
        with self.assertRaises(UnsupportedFeatureError):
            await model.connect(
                RealtimeSessionConfig(
                    tools={"lookup": ToolDefinition(name="lookup", description=None, schema={"type": "object"})}
                )
            )

        session = await model.connect(RealtimeSessionConfig(translation_target_language_code="es"))
        with self.assertRaises(UnsupportedFeatureError):
            await session.send_text("hello")
        await session.aclose()

    async def test_gemini_live_translate_browser_token_includes_constraints(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any] | None,
            timeout_ms: int | None,
            method: str = "POST",
            body: Any = None,
            stream: bool = False,
        ):
            requests.append({"url": url, "headers": headers, "json": json_body})

            class FakeResponse:
                status_code = 200
                headers = {"content-type": "application/json"}

                async def json(self) -> Any:
                    return {"authToken": {"name": "authTokens/live-translate"}}

                async def text(self) -> str:
                    return json.dumps(await self.json())

                async def read(self) -> bytes:
                    return json.dumps(await self.json()).encode()

                async def iter_lines(self):
                    if False:
                        yield ""

            return FakeResponse()

        provider = create_gemini(api_key="test", fetch=fetch)
        token = await provider.realtime_model("gemini-3.5-live-translate-preview").create_browser_token(
            RealtimeSessionConfig(
                translation_target_language_code="pl",
                translation_echo_target_language=True,
                provider_options={"uses": 1},
            )
        )

        self.assertEqual(token.value, "authTokens/live-translate")
        auth_token = requests[0]["json"]["authToken"]
        self.assertEqual(auth_token["uses"], 1)
        constraints = auth_token["liveConnectConstraints"]
        self.assertEqual(constraints["model"], "gemini-3.5-live-translate-preview")
        self.assertEqual(constraints["config"]["responseModalities"], ["AUDIO"])
        self.assertEqual(constraints["config"]["inputAudioTranscription"], {})
        self.assertEqual(constraints["config"]["outputAudioTranscription"], {})
        self.assertEqual(
            constraints["config"]["translationConfig"],
            {"targetLanguageCode": "pl", "echoTargetLanguage": True},
        )


class FakeLiveSession:
    def __init__(self, events: list[Any] | None = None) -> None:
        self.sent_audio: list[AudioFrame] = []
        self.sent_text: list[str] = []
        self.sent_tool_results: list[ToolExecutionResult] = []
        self.updated: list[dict[str, Any]] = []
        self.closed = False
        self.events = events

    async def send_audio(self, frame: AudioFrame) -> None:
        self.sent_audio.append(frame)

    async def send_text(self, text: str) -> None:
        self.sent_text.append(text)

    async def send_tool_result(self, result: ToolExecutionResult) -> None:
        self.sent_tool_results.append(result)

    async def update(self, **kwargs: Any) -> None:
        self.updated.append(kwargs)

    def event_stream(self):
        async def generator():
            events = self.events or [
                RealtimeTranscriptEvent(text="buscar clima", role="user", is_final=True),
                RealtimeToolCallEvent(tool_call=ToolCall(id="call_1", name="weather", input={"city": "BA"})),
                RealtimeTextDeltaEvent(text_delta="Hace sol"),
                RealtimeTranscriptEvent(text="Hace sol", role="assistant", is_final=True),
                RealtimeResponseCompletedEvent(reason="done"),
            ]
            for event in events:
                if isinstance(event, BaseException):
                    raise event
                yield event

        return generator()

    async def aclose(self) -> None:
        self.closed = True


class FakeLiveModel:
    provider = "openai"
    model_id = "gpt-realtime-2.1"
    capabilities = ModelCapabilities(
        streaming=False,
        tools=True,
        structured_output=False,
        json_mode=False,
        tool_choice=True,
        parallel_tool_calls=False,
        vision=False,
        files=False,
        audio_input=True,
        audio_output=True,
        embeddings=False,
        reasoning=False,
        web_search=False,
        realtime=True,
        realtime_audio_input=True,
        realtime_audio_output=True,
        realtime_tools=True,
    )

    def __init__(self, session: FakeLiveSession) -> None:
        self._session = session
        self.connect_calls = 0
        self.generate_calls = 0
        self.last_config: RealtimeSessionConfig | None = None

    async def connect(self, config: RealtimeSessionConfig | None = None, options: RealtimeConnectOptions | None = None):
        self.connect_calls += 1
        self.last_config = config
        return self._session

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        self.generate_calls += 1
        return GenerateResult(
            messages=[create_text_message("assistant", "Aprobación resuelta")],
            text="Aprobación resuelta",
            finish_reason="stop",
        )

    async def create_browser_token(self, config: RealtimeSessionConfig | None = None, options: RealtimeConnectOptions | None = None):
        raise RuntimeError("not used")


class LiveAgentTests(IsolatedAsyncioTestCase):
    async def test_stream_live_agent_executes_tools_and_persists_transcript(self) -> None:
        live_session = FakeLiveSession()
        agent: Agent[Any, Any] = Agent(
            name="assistant",
            model=FakeLiveModel(live_session),
            instructions="Be helpful.",
            memory=create_in_memory_agent_memory_store(),
            tools={
                "weather": ToolDefinition(
                    name="weather",
                    description="Weather lookup",
                    schema={"type": "object"},
                    execute=lambda payload: {"forecast": f"sunny in {payload['city']}"},
                )
            },
        )

        stream = stream_live_agent(agent=agent)
        events = [event async for event in stream.event_stream()]
        result = await stream.collect()

        self.assertTrue(any(isinstance(event, AgentTextDeltaEvent) and event.text_delta == "Hace sol" for event in events))
        self.assertTrue(any(isinstance(event, RealtimeToolResultEvent) for event in events))
        self.assertEqual(live_session.sent_tool_results[0].output, {"forecast": "sunny in BA"})
        self.assertEqual(result.text, "Hace sol")
        final_part = result.session.messages[-1].parts[0]
        self.assertIsInstance(final_part, TextPart)
        self.assertEqual(final_part.text, "Hace sol")  # type: ignore[union-attr]

    async def test_stream_live_agent_merges_agent_tools_into_explicit_realtime_config(self) -> None:
        live_session = FakeLiveSession()
        model = FakeLiveModel(live_session)
        agent: Agent[Any, Any] = Agent(
            name="assistant",
            model=model,
            tools={
                "weather": ToolDefinition(
                    name="weather",
                    description="Weather lookup",
                    schema={"type": "object"},
                    execute=lambda payload: {"forecast": f"sunny in {payload['city']}"},
                )
            },
        )

        result = await stream_live_agent(
            agent=agent,
            realtime_config=RealtimeSessionConfig(voice="alloy"),
        ).collect()

        self.assertEqual(result.text, "Hace sol")
        self.assertIsNotNone(model.last_config)
        self.assertIn("weather", model.last_config.tools or {})
        self.assertEqual(model.last_config.voice, "alloy")

    async def test_stream_live_agent_persists_suspension_and_can_resume(self) -> None:
        live_session = FakeLiveSession(
            [
                RealtimeTranscriptEvent(text="buscar clima", role="user", is_final=True),
                RealtimeToolCallEvent(tool_call=ToolCall(id="call_approval", name="weather", input={"city": "BA"})),
            ]
        )
        model = FakeLiveModel(live_session)
        run_store = create_in_memory_agent_run_store()
        executions: list[dict[str, Any]] = []

        class SuspendPolicy:
            async def __call__(self, _request: ToolApprovalRequest) -> ApprovalDecision:
                return ApprovalDecision.require_human("Confirmar consulta externa.")

        def weather(payload: dict[str, Any]) -> dict[str, Any]:
            executions.append(payload)
            return {"forecast": "sunny"}

        agent: Agent[Any, Any] = Agent(
            name="assistant",
            model=model,
            run_store=run_store,
            approval_policy=SuspendPolicy(),  # type: ignore[arg-type]
            tools={
                "weather": ToolDefinition(
                    name="weather",
                    description="Weather lookup",
                    schema={"type": "object"},
                    execute=weather,
                    requires_approval=True,
                )
            },
        )

        suspended = await stream_live_agent(agent=agent, idempotency_key="live:approval").collect()

        self.assertEqual(suspended.state.status, "suspended")  # type: ignore[union-attr]
        self.assertEqual(len(suspended.state.pending_approvals), 1)  # type: ignore[union-attr]
        self.assertEqual(executions, [])
        stored = await run_store.load(suspended.run_id)
        self.assertEqual(stored.status, "suspended")  # type: ignore[union-attr]
        self.assertIn("resume_messages", stored.metadata)  # type: ignore[union-attr]

        resumed = await resume_agent_run(agent=agent, run_id=suspended.run_id)

        self.assertEqual(resumed.text, "Aprobación resuelta")
        self.assertEqual(executions, [{"city": "BA"}])
        final_state = await run_store.load(suspended.run_id)
        self.assertEqual(final_state.status, "completed")  # type: ignore[union-attr]

    async def test_stream_live_agent_reuses_completed_idempotent_run(self) -> None:
        live_session = FakeLiveSession(
            [
                RealtimeTranscriptEvent(text="listo", role="assistant", is_final=True),
                RealtimeResponseCompletedEvent(reason="done"),
            ]
        )
        model = FakeLiveModel(live_session)
        run_store = create_in_memory_agent_run_store()
        agent: Agent[Any, Any] = Agent(name="assistant", model=model, run_store=run_store)

        first = await stream_live_agent(agent=agent, idempotency_key="live:completed").collect()
        second = await stream_live_agent(agent=agent, idempotency_key="live:completed").collect()

        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(second.text, "listo")
        self.assertEqual(second.state.status, "completed")  # type: ignore[union-attr]
        self.assertEqual(model.connect_calls, 1)

    async def test_stream_live_agent_observes_durable_cancellation(self) -> None:
        release = asyncio.Event()

        class BlockingLiveSession(FakeLiveSession):
            def event_stream(self):
                async def generator():
                    await release.wait()
                    if False:
                        yield RealtimeResponseCompletedEvent(reason="done")

                return generator()

        live_session = BlockingLiveSession()
        run_store = create_in_memory_agent_run_store()
        agent: Agent[Any, Any] = Agent(name="assistant", model=FakeLiveModel(live_session), run_store=run_store)
        stream = stream_live_agent(agent=agent, idempotency_key="live:cancel")

        state = None
        for _ in range(20):
            state = await run_store.find_by_idempotency_key("live:cancel")
            if state is not None:
                break
            await asyncio.sleep(0)
        self.assertIsNotNone(state)
        await cancel_agent_run(run_store, state.run_id, reason="operator stop")  # type: ignore[union-attr]

        with self.assertRaises(AgentRunCancelled):
            await stream.collect()

        stored = await run_store.load(state.run_id)  # type: ignore[union-attr]
        self.assertEqual(stored.status, "cancelled")  # type: ignore[union-attr]
        self.assertEqual(stored.cancellation_reason, "operator stop")  # type: ignore[union-attr]
        self.assertTrue(live_session.closed)

    async def test_stream_live_agent_observes_in_process_cancellation_token(self) -> None:
        release = asyncio.Event()

        class BlockingLiveSession(FakeLiveSession):
            def event_stream(self):
                async def generator():
                    await release.wait()
                    if False:
                        yield RealtimeResponseCompletedEvent(reason="done")

                return generator()

        live_session = BlockingLiveSession()
        token = AgentCancellationToken()
        model = FakeLiveModel(live_session)
        agent: Agent[Any, Any] = Agent(name="assistant", model=model)
        stream = stream_live_agent(agent=agent, cancellation_token=token)
        for _ in range(20):
            if model.connect_calls:
                break
            await asyncio.sleep(0)
        self.assertEqual(model.connect_calls, 1)
        await stream.send_text("ready")
        token.cancel("request disconnected")

        with self.assertRaisesRegex(AgentRunCancelled, "request disconnected"):
            await stream.collect()

        self.assertTrue(live_session.closed)

    async def test_stream_live_agent_persists_failures(self) -> None:
        live_session = FakeLiveSession([RuntimeError("realtime transport failed")])
        run_store = create_in_memory_agent_run_store()
        agent: Agent[Any, Any] = Agent(name="assistant", model=FakeLiveModel(live_session), run_store=run_store)
        stream = stream_live_agent(agent=agent, idempotency_key="live:error")

        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            await stream.collect()

        state = await run_store.find_by_idempotency_key("live:error")
        self.assertIsNotNone(state)
        self.assertEqual(state.status, "failed")  # type: ignore[union-attr]
        self.assertIn("transport failed", state.error or "")  # type: ignore[union-attr]
        self.assertTrue(live_session.closed)

    async def test_stream_live_agent_preserves_unknown_outcome_on_tool_timeout(self) -> None:
        live_session = FakeLiveSession(
            [RealtimeToolCallEvent(tool_call=ToolCall(id="call_slow", name="slow", input={}))]
        )
        run_store = create_in_memory_agent_run_store()

        async def slow_tool(_payload: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(1)
            return {"ok": True}

        agent: Agent[Any, Any] = Agent(
            name="assistant",
            model=FakeLiveModel(live_session),
            run_store=run_store,
            tools={
                "slow": ToolDefinition(
                    name="slow",
                    description="Slow tool",
                    schema={"type": "object"},
                    execute=slow_tool,
                )
            },
        )

        with self.assertRaises(ToolExecutionOutcomeUnknown):
            await stream_live_agent(
                agent=agent,
                idempotency_key="live:timeout",
                tool_execution=ToolExecutionOptions(timeout_ms=10),
            ).collect()

        state = await run_store.find_by_idempotency_key("live:timeout")
        self.assertIsNotNone(state)
        self.assertEqual(state.status, "failed")  # type: ignore[union-attr]
        self.assertIn("external outcome is unknown", state.error or "")  # type: ignore[union-attr]
