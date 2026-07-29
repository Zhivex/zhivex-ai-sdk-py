from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any
from unittest import IsolatedAsyncioTestCase

from pydantic import BaseModel

from zhivex_ai import (
    Agent,
    AgentContext,
    AgentHooks,
    AgentRunRequest,
    ApprovalDecision,
    ModelCapabilities,
    ModelMessage,
    ParseError,
    ToolDefinition,
    ToolExecutionContext,
    create_in_memory_agent_run_store,
    create_text_message,
    handoff_to,
    run_agent,
    stream_agent,
)
from zhivex_ai.types import (
    GenerateResult,
    ModelGenerateInput,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    ToolCall,
    ToolCallPart,
)


def capabilities(*, structured_output: bool = True) -> ModelCapabilities:
    return ModelCapabilities(
        streaming=True,
        tools=True,
        structured_output=structured_output,
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


class JsonModel:
    provider = "test"
    model_id = "json"

    def __init__(self, text: str, *, structured_output: bool = True) -> None:
        self.text = text
        self.capabilities = capabilities(structured_output=structured_output)
        self.inputs: list[ModelGenerateInput] = []

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        self.inputs.append(input)
        return GenerateResult(
            messages=[create_text_message("assistant", self.text)],
            text=self.text,
            finish_reason="stop",
        )

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        self.inputs.append(input)

        async def generator() -> AsyncIterable[object]:
            yield StreamTextDeltaEvent(text_delta=self.text)
            yield StreamFinishEvent(finish_reason="stop")

        return generator()


class ToolLoopJsonModel(JsonModel):
    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        self.inputs.append(input)
        if not any(message.role == "tool" for message in input.messages):
            return GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[ToolCallPart(tool_call=ToolCall(id="call_1", name="lookup", input={"id": "42"}))],
                    )
                ],
                finish_reason="tool-calls",
            )
        return GenerateResult(
            messages=[create_text_message("assistant", self.text)],
            text=self.text,
            finish_reason="stop",
        )


class HandoffModel(JsonModel):
    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        self.inputs.append(input)
        if not any(message.role == "tool" for message in input.messages):
            return GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[ToolCallPart(tool_call=ToolCall(id="handoff_1", name="delegate", input={}))],
                    )
                ],
                finish_reason="tool-calls",
            )
        return GenerateResult(
            messages=[create_text_message("assistant", "delegated")],
            text="delegated",
            finish_reason="stop",
        )


@dataclass
class Dependencies:
    tenant: str
    secret: str


class Answer(BaseModel):
    answer: str
    confidence: float


class OtherAnswer(BaseModel):
    ignored: bool


class RecordingHooks(AgentHooks):
    def __init__(self, label: str, events: list[str]) -> None:
        self.label = label
        self.events = events

    async def on_agent_start(self, context: AgentContext[Any], agent: Agent[Any, Any]) -> None:
        self.events.append(f"{self.label}:agent-start:{agent.name}")

    async def on_agent_end(self, context: AgentContext[Any], agent: Agent[Any, Any], result: Any) -> None:
        self.events.append(f"{self.label}:agent-end:{agent.name}")

    async def on_model_start(self, context: AgentContext[Any], agent: Agent[Any, Any], input: Any) -> None:
        self.events.append(f"{self.label}:model-start")

    async def on_model_end(self, context: AgentContext[Any], agent: Agent[Any, Any], result: Any) -> None:
        self.events.append(f"{self.label}:model-end")

    async def on_tool_start(self, context: AgentContext[Any], agent: Agent[Any, Any], *args: Any) -> None:
        self.events.append(f"{self.label}:tool-start")

    async def on_tool_end(self, context: AgentContext[Any], agent: Agent[Any, Any], *args: Any) -> None:
        self.events.append(f"{self.label}:tool-end")

    async def on_handoff(self, context: AgentContext[Any], source: Any, target: Any, handoff: Any) -> None:
        self.events.append(f"{self.label}:handoff:{source.name}->{target.name}")


class AgentDxTests(IsolatedAsyncioTestCase):
    async def test_typed_deps_dynamic_instructions_and_native_output(self) -> None:
        model = JsonModel('{"answer":"approved","confidence":0.98}')
        deps = Dependencies(tenant="bank-ar", secret="do-not-log")
        calls: list[AgentContext[Dependencies]] = []

        async def instructions(context: AgentContext[Dependencies]) -> str:
            calls.append(context)
            return f"Serve tenant {context.deps.tenant}."  # type: ignore[union-attr]

        agent: Agent[Dependencies, Answer] = Agent(
            name="analyst",
            model=model,
            instructions=instructions,
            output_type=Answer,
        )
        result = await run_agent(agent=agent, prompt="Decide", deps=deps)

        self.assertEqual(result.text, '{"answer":"approved","confidence":0.98}')
        self.assertEqual(result.output, Answer(answer="approved", confidence=0.98))
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0].deps, deps)
        self.assertIs(calls[0].session, result.session)
        self.assertNotIn(deps.secret, repr(calls[0]))
        self.assertEqual(model.inputs[0].structured_output.schema, Answer)  # type: ignore[union-attr]
        system_text = "\n".join(
            part.text
            for message in model.inputs[0].messages
            if message.role == "system"
            for part in message.parts
            if part.type == "text"
        )
        self.assertIn("Serve tenant bank-ar.", system_text)
        self.assertNotIn("Serve tenant bank-ar.", " ".join(map(str, result.session.messages)))

    async def test_prompted_output_stream_collects_typed_result(self) -> None:
        model = JsonModel('{"answer":"streamed","confidence":0.75}', structured_output=False)
        agent: Agent[None, Answer] = Agent(
            name="streamer",
            model=model,
            output_type=Answer,
        )

        result = await stream_agent(agent=agent, prompt="Answer").collect()

        self.assertEqual(result.output.answer, "streamed")  # type: ignore[union-attr]
        self.assertIsNone(model.inputs[0].structured_output)
        system_text = "\n".join(
            part.text
            for message in model.inputs[0].messages
            if message.role == "system"
            for part in message.parts
            if part.type == "text"
        )
        self.assertIn("JSON Schema", system_text)
        self.assertIn('"confidence"', system_text)

    async def test_hooks_cover_each_model_call_and_tool_with_nested_order(self) -> None:
        events: list[str] = []
        deps = Dependencies(tenant="bank-ar", secret="hidden")
        seen_tool_contexts: list[ToolExecutionContext[Dependencies]] = []

        async def lookup(input: dict[str, str], context: ToolExecutionContext[Dependencies]) -> dict[str, str]:
            seen_tool_contexts.append(context)
            return {"value": context.deps.tenant}  # type: ignore[union-attr]

        model = ToolLoopJsonModel('{"answer":"done","confidence":1.0}')
        agent: Agent[Dependencies, Answer] = Agent(
            name="tool-agent",
            model=model,
            tools={
                "lookup": ToolDefinition(
                    name="lookup",
                    description="Lookup",
                    schema={"type": "object"},
                    execute=lookup,
                )
            },
            output_type=Answer,
            hooks=[RecordingHooks("agent", events)],
        )
        result = await run_agent(
            agent=agent,
            prompt="Lookup",
            deps=deps,
            hooks=[RecordingHooks("run", events)],
        )

        self.assertEqual(result.output.answer, "done")  # type: ignore[union-attr]
        self.assertIs(seen_tool_contexts[0].deps, deps)
        self.assertNotIn(deps.secret, repr(seen_tool_contexts[0]))
        self.assertEqual(
            events,
            [
                "run:agent-start:tool-agent",
                "agent:agent-start:tool-agent",
                "run:model-start",
                "agent:model-start",
                "agent:model-end",
                "run:model-end",
                "run:tool-start",
                "agent:tool-start",
                "agent:tool-end",
                "run:tool-end",
                "run:model-start",
                "agent:model-start",
                "agent:model-end",
                "run:model-end",
                "agent:agent-end:tool-agent",
                "run:agent-end:tool-agent",
            ],
        )

    async def test_middleware_can_transform_request_with_outer_order(self) -> None:
        model = JsonModel("legacy")
        events: list[str] = []

        async def outer(request: AgentRunRequest[Any, Any], call_next: Any) -> Any:
            events.append("outer:start")
            request.prompt = "changed"
            result = await call_next(request)
            events.append("outer:end")
            return result

        async def inner(request: AgentRunRequest[Any, Any], call_next: Any) -> Any:
            events.append("inner:start")
            result = await call_next(request)
            events.append("inner:end")
            return result

        agent: Agent[Any, str] = Agent(name="middleware", model=model, middleware=[inner])
        result = await run_agent(agent=agent, prompt="original", middleware=[outer])

        self.assertEqual(result.output, "legacy")
        self.assertEqual(events, ["outer:start", "inner:start", "inner:end", "outer:end"])
        user_text = next(
            part.text
            for message in model.inputs[0].messages
            if message.role == "user"
            for part in message.parts
            if part.type == "text"
        )
        self.assertEqual(user_text, "changed")

    async def test_root_output_contract_governs_handoff_result(self) -> None:
        source_model = HandoffModel("unused")
        child_model = JsonModel('{"answer":"from-child","confidence":0.9}')
        child: Agent[Dependencies, OtherAnswer] = Agent(
            name="child",
            model=child_model,
            output_type=OtherAnswer,
        )
        source: Agent[Dependencies, Answer] = Agent(
            name="source",
            model=source_model,
            tools={
                "delegate": ToolDefinition(
                    name="delegate",
                    description="Delegate",
                    schema={"type": "object"},
                    execute=lambda _input: handoff_to("child"),
                )
            },
            subagents={"child": child},
            output_type=Answer,
        )

        result = await run_agent(
            agent=source,
            prompt="Delegate",
            deps=Dependencies(tenant="bank-ar", secret="hidden"),
        )

        self.assertEqual(result.output, Answer(answer="from-child", confidence=0.9))
        self.assertEqual(result.agent_name, "child")
        self.assertEqual(child_model.inputs[0].structured_output.schema, Answer)  # type: ignore[union-attr]

    async def test_idempotent_reuse_rehydrates_typed_output(self) -> None:
        model = JsonModel('{"answer":"cached","confidence":0.8}')
        agent: Agent[None, Answer] = Agent(
            name="cached",
            model=model,
            output_type=Answer,
            run_store=create_in_memory_agent_run_store(),
        )

        first = await run_agent(agent=agent, prompt="Answer", idempotency_key="same")
        second = await run_agent(agent=agent, prompt="Ignored", idempotency_key="same")

        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(second.output, Answer(answer="cached", confidence=0.8))
        self.assertEqual(len(model.inputs), 1)

    async def test_invalid_typed_output_and_dynamic_instruction_return_fail_closed(self) -> None:
        invalid_output_agent: Agent[None, Answer] = Agent(
            name="invalid-output",
            model=JsonModel("not-json"),
            output_type=Answer,
        )
        with self.assertRaises(ParseError):
            await run_agent(agent=invalid_output_agent, prompt="Answer")

        def invalid_instructions(_context: AgentContext[None]) -> Any:
            return 42

        invalid_instruction_agent: Agent[None, str] = Agent(
            name="invalid-instructions",
            model=JsonModel("unused"),
            instructions=invalid_instructions,
        )
        with self.assertRaisesRegex(TypeError, "must return str or None"):
            await run_agent(agent=invalid_instruction_agent, prompt="Answer")

    async def test_suspended_typed_run_has_no_output(self) -> None:
        model = ToolLoopJsonModel('{"answer":"never","confidence":1.0}')

        async def approval(_request: Any) -> ApprovalDecision:
            return ApprovalDecision.require_human("operator review")

        agent: Agent[None, Answer] = Agent(
            name="suspended",
            model=model,
            tools={
                "lookup": ToolDefinition(
                    name="lookup",
                    description="Lookup",
                    schema={"type": "object"},
                    execute=lambda _input: {"ok": True},
                    requires_approval=True,
                )
            },
            approval_policy=approval,
            output_type=Answer,
            run_store=create_in_memory_agent_run_store(),
        )

        result = await run_agent(agent=agent, prompt="Lookup")

        self.assertIsNone(result.output)
        self.assertEqual(result.state.status, "suspended")  # type: ignore[union-attr]
