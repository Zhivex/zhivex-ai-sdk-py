from __future__ import annotations

from collections.abc import AsyncIterable
import sys
import types
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    Agent,
    AgentRegistry,
    AgentToolApprovalEvent,
    AgentToolCallEvent,
    ModelCapabilities,
    ModelMessage,
    SummaryConfig,
    ToolExecutionContext,
    create_agent_session,
    create_in_memory_agent_memory_store,
    create_in_memory_checkpoint_store,
    create_otel_agent_observer,
    create_text_message,
    handoff_to,
    load_agent_session,
    permission_allowlist_approval_policy,
    run_agent,
    stream_agent,
    tool,
)
from zhivex_ai.observability import OTelAgentObserver  # noqa: E402
from zhivex_ai.types import (  # noqa: E402
    GenerateResult,
    ModelGenerateInput,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    ToolCall,
    ToolCallPart,
    TokenUsage,
)


BASE_CAPABILITIES = ModelCapabilities(
    streaming=True,
    tools=True,
    structured_output=True,
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


class EchoAgentModel:
    provider = "test"
    model_id = "echo"
    capabilities = BASE_CAPABILITIES

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        last_user = next((message for message in reversed(input.messages) if message.role == "user"), None)
        text = f"echo:{''.join(part.text for part in last_user.parts if part.type == 'text')}" if last_user else "echo:"
        return GenerateResult(messages=[create_text_message("assistant", text)], text=text)

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        async def generator() -> AsyncIterable[object]:
            yield StreamTextDeltaEvent(text_delta="echo")
            yield StreamFinishEvent(
                finish_reason="stop",
                usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        return generator()


class ToolLoopModel:
    provider = "test"
    model_id = "tool-loop"
    capabilities = BASE_CAPABILITIES

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        has_tool_message = any(message.role == "tool" for message in input.messages)
        if not has_tool_message and input.tools:
            return GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[ToolCallPart(tool_call=ToolCall(id="call_1", name="delegate", input={"task": "research"}))],
                    )
                ]
            )
        return GenerateResult(messages=[create_text_message("assistant", "handoff-complete")], text="handoff-complete")

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        async def generator() -> AsyncIterable[object]:
            yield StreamTextDeltaEvent(text_delta="handoff-complete")
            yield StreamFinishEvent(
                finish_reason="stop",
                usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        return generator()


class PermissionToolModel:
    provider = "test"
    model_id = "permissions"
    capabilities = BASE_CAPABILITIES

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        has_tool_message = any(message.role == "tool" for message in input.messages)
        if not has_tool_message:
            return GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[ToolCallPart(tool_call=ToolCall(id="call_1", name="secret_lookup", input={"item": "apollo"}))],
                    )
                ]
            )
        return GenerateResult(messages=[create_text_message("assistant", "done")], text="done")

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        async def generator() -> AsyncIterable[object]:
            yield StreamTextDeltaEvent(text_delta="done")
            yield StreamFinishEvent(
                finish_reason="stop",
                usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        return generator()


class AgentRuntimeTests(IsolatedAsyncioTestCase):
    async def test_run_agent_persists_session_memory_and_summary(self) -> None:
        memory = create_in_memory_agent_memory_store(
            summary_config=SummaryConfig(max_messages=2, preserve_recent_messages=1, max_summary_chars=200)
        )
        agent = Agent(
            name="assistant",
            instructions="Be concise.",
            model=EchoAgentModel(),
            memory=memory,
        )

        session = create_agent_session()
        await run_agent(agent=agent, session=session, prompt="hello")
        second = await run_agent(agent=agent, session=session, prompt="again")

        self.assertEqual(second.text, "echo:again")
        restored = await load_agent_session(agent, session.id)
        self.assertIsNotNone(restored.summary)
        self.assertIn("user: hello", restored.summary or "")

    async def test_run_agent_executes_handoff_via_subagent(self) -> None:
        agent = Agent(
            name="triage",
            instructions="Delegate when needed.",
            model=ToolLoopModel(),
            tools={
                "delegate": tool(
                    name="delegate",
                    schema=dict[str, str],
                    execute=lambda input: handoff_to("researcher", input="find background context"),
                )
            },
            subagents={
                "researcher": Agent(
                    name="researcher",
                    instructions="Answer directly.",
                    model=EchoAgentModel(),
                )
            },
        )

        result = await run_agent(agent=agent, prompt="help me plan", max_steps=2)

        self.assertEqual(result.text, "echo:find background context")
        self.assertEqual(result.orchestration_path, ["triage", "researcher"])
        self.assertEqual(result.trace.handoff_count, 1)

    async def test_run_agent_resolves_registry_handoffs(self) -> None:
        runtime_registry = AgentRegistry(
            {
                "researcher": Agent(
                    name="researcher",
                    instructions="Answer directly.",
                    model=EchoAgentModel(),
                )
            }
        )
        agent = Agent(
            name="triage",
            instructions="Delegate when needed.",
            model=ToolLoopModel(),
            tools={
                "delegate": tool(
                    name="delegate",
                    schema=dict[str, str],
                    execute=lambda input: handoff_to("researcher", input="registry task"),
                )
            },
        )

        result = await run_agent(agent=agent, prompt="help me plan", registry=runtime_registry, max_steps=2)

        self.assertEqual(result.text, "echo:registry task")
        self.assertEqual(result.orchestration_path, ["triage", "researcher"])

    async def test_run_agent_can_stop_on_handoff(self) -> None:
        agent = Agent(
            name="triage",
            instructions="Delegate when needed.",
            model=ToolLoopModel(),
            tools={
                "delegate": tool(
                    name="delegate",
                    schema=dict[str, str],
                    execute=lambda input: handoff_to("researcher", input="find background context"),
                )
            },
        )

        result = await run_agent(agent=agent, prompt="help me plan", max_steps=2, stop_on_handoff=True)

        self.assertIsNotNone(result.handoff)
        self.assertEqual(result.handoff.target_agent, "researcher")
        self.assertEqual(result.text, "handoff-complete")

    async def test_run_agent_applies_permission_approval_policy(self) -> None:
        observed_contexts: list[ToolExecutionContext] = []

        async def execute_secret(input: dict[str, str], context: ToolExecutionContext) -> dict[str, str]:
            observed_contexts.append(context)
            return {"item": input["item"], "status": "ok"}

        agent = Agent(
            name="assistant",
            instructions="Use tools when needed.",
            model=PermissionToolModel(),
            tools={
                "secret_lookup": tool(
                    name="secret_lookup",
                    schema=dict[str, str],
                    execute=execute_secret,
                    permissions=["project:read"],
                    requires_approval=True,
                )
            },
            approval_policy=permission_allowlist_approval_policy("billing:read"),
        )

        result = await run_agent(agent=agent, prompt="help me plan", max_steps=2)

        self.assertTrue(result.tool_results[0].is_error)
        approvals = [event for event in result.trace.events if isinstance(event, AgentToolApprovalEvent)]
        self.assertEqual(len(approvals), 1)
        self.assertFalse(approvals[0].approved)
        self.assertEqual(observed_contexts, [])

    async def test_run_agent_saves_checkpoints(self) -> None:
        checkpoints = create_in_memory_checkpoint_store()
        agent = Agent(
            name="assistant",
            instructions="Use tools when needed.",
            model=PermissionToolModel(),
            tools={
                "secret_lookup": tool(
                    name="secret_lookup",
                    schema=dict[str, str],
                    execute=lambda input: {"accepted": True},
                )
            },
            checkpoint_store=checkpoints,
        )

        result = await run_agent(agent=agent, prompt="help me plan", max_steps=2)
        saved = await checkpoints.list(run_id=result.run_id)

        self.assertEqual(len(saved), 2)
        self.assertTrue(saved[-1].is_final)

    async def test_stream_agent_emits_events_and_collects(self) -> None:
        agent = Agent(
            name="assistant",
            instructions="Be concise.",
            model=EchoAgentModel(),
        )

        stream = stream_agent(agent=agent, prompt="hello")
        event_types: list[str] = []
        async for event in stream.event_stream():
            event_types.append(event.type)

        final = await stream.collect()

        self.assertIn("run-start", event_types)
        self.assertIn("delegation-start", event_types)
        self.assertIn("text-delta", event_types)
        self.assertIn("finish", event_types)
        self.assertEqual(final.text, "echo")

    async def test_run_agent_emits_tool_call_events(self) -> None:
        agent = Agent(
            name="assistant",
            instructions="Use tools when needed.",
            model=PermissionToolModel(),
            tools={
                "secret_lookup": tool(
                    name="secret_lookup",
                    schema=dict[str, str],
                    execute=lambda input: {"accepted": True},
                )
            },
        )

        result = await run_agent(agent=agent, prompt="help me plan", max_steps=2)
        tool_calls = [event for event in result.trace.events if isinstance(event, AgentToolCallEvent)]

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].tool_call.name, "secret_lookup")

    async def test_create_otel_agent_observer_uses_tracer(self) -> None:
        class FakeSpan:
            def __init__(self) -> None:
                self.attributes: dict[str, object] = {}
                self.exceptions: list[Exception] = []

            def set_attribute(self, key: str, value: object) -> None:
                self.attributes[key] = value

            def record_exception(self, error: Exception) -> None:
                self.exceptions.append(error)

            def set_status(self, status: object) -> None:
                self.status = status

        class FakeManager:
            def __init__(self, span: FakeSpan) -> None:
                self.span = span
                self.closed = False

            def __enter__(self) -> FakeSpan:
                return self.span

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                self.closed = True

        class FakeTracer:
            def __init__(self) -> None:
                self.started: list[tuple[str, FakeSpan]] = []

            def start_as_current_span(self, name: str) -> FakeManager:
                span = FakeSpan()
                self.started.append((name, span))
                return FakeManager(span)

        fake_trace_module = types.ModuleType("opentelemetry.trace")
        fake_trace_module.StatusCode = types.SimpleNamespace(ERROR="ERROR")
        fake_trace_module.Status = lambda code, description=None: (code, description)

        fake_root_module = types.ModuleType("opentelemetry")
        tracer = FakeTracer()
        fake_root_module.trace = types.SimpleNamespace(get_tracer=lambda name, version=None: tracer)

        previous_root = sys.modules.get("opentelemetry")
        previous_trace = sys.modules.get("opentelemetry.trace")
        sys.modules["opentelemetry"] = fake_root_module
        sys.modules["opentelemetry.trace"] = fake_trace_module
        try:
            observer = create_otel_agent_observer()
            self.assertIsInstance(observer, OTelAgentObserver)
            handle = observer.start_span("demo", {"agent.name": "assistant"})
            handle.end(attributes={"finish.reason": "stop"})
        finally:
            if previous_root is not None:
                sys.modules["opentelemetry"] = previous_root
            else:
                sys.modules.pop("opentelemetry", None)
            if previous_trace is not None:
                sys.modules["opentelemetry.trace"] = previous_trace
            else:
                sys.modules.pop("opentelemetry.trace", None)

        self.assertEqual(tracer.started[0][0], "demo")
        self.assertEqual(tracer.started[0][1].attributes["agent.name"], "assistant")
        self.assertEqual(tracer.started[0][1].attributes["finish.reason"], "stop")
