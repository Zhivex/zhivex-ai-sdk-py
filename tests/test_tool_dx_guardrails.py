from __future__ import annotations

import sys
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import Agent, ModelCapabilities, run_agent  # noqa: E402
from zhivex_ai.agent import AgentGuardrailEvent, LocalToolRuntime, ToolRegistry  # noqa: E402
from zhivex_ai.messages import create_text_message, tool  # noqa: E402
from zhivex_ai.schema import create_schema_adapter  # noqa: E402
from zhivex_ai.types import (  # noqa: E402
    GenerateResult,
    MCPServerConfig,
    MCPToolConfig,
    ModelGenerateInput,
    ModelMessage,
    RemoteHTTPToolConfig,
    ToolCall,
    ToolCallPart,
    ToolDefinition,
    ToolExecutionContext,
    ToolGuardrailResult,
)


CAPABILITIES = ModelCapabilities(
    streaming=False,
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


class ToolCallingModel:
    provider = "test"
    model_id = "tool-calling"
    capabilities = CAPABILITIES

    def __init__(self, tool_name: str = "lookup") -> None:
        self.tool_name = tool_name

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        if not any(message.role == "tool" for message in input.messages):
            return GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[
                            ToolCallPart(
                                tool_call=ToolCall(
                                    id="call_1",
                                    name=self.tool_name,
                                    input={"item": "apollo"},
                                )
                            )
                        ],
                    )
                ],
                finish_reason="tool-calls",
            )
        return GenerateResult(messages=[create_text_message("assistant", "done")], text="done")


class RecordingToolRuntime:
    def __init__(self, output: object | None = None) -> None:
        self.calls = 0
        self.output = {"secret": "raw-secret", "safe": False} if output is None else output
        self.last_input = None

    async def execute(self, definition, input, context):
        self.calls += 1
        self.last_input = input
        return self.output

    async def aclose(self) -> None:
        return None


class ToolDecoratorTests(IsolatedAsyncioTestCase):
    async def test_bare_decorator_derives_schema_docstring_and_executes_sync_callable(self) -> None:
        @tool
        def weather(city: str, days: int = 1) -> dict[str, object]:
            """Get a weather forecast."""

            return {"city": city, "days": days}

        schema = create_schema_adapter(weather.schema).json_schema()
        parsed = create_schema_adapter(weather.schema).validate_python({"city": "Madrid", "days": 3})
        result = await LocalToolRuntime().execute(
            weather,
            parsed,
            ToolExecutionContext(tool_name="weather"),
        )

        self.assertEqual(weather.name, "weather")
        self.assertEqual(weather.description, "Get a weather forecast.")
        self.assertEqual(schema["properties"]["city"]["type"], "string")
        self.assertEqual(schema["properties"]["days"]["default"], 1)
        self.assertEqual(schema["required"], ["city"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(result, {"city": "Madrid", "days": 3})

    async def test_configured_decorator_supports_async_callable_and_context(self) -> None:
        @tool(name="project_lookup", description="Find a project.")
        async def lookup(item: str, context: ToolExecutionContext) -> str:
            return f"{context.tool_name}:{item}"

        schema = create_schema_adapter(lookup.schema).json_schema()
        parsed = create_schema_adapter(lookup.schema).validate_python({"item": "apollo"})
        result = await LocalToolRuntime().execute(
            lookup,
            parsed,
            ToolExecutionContext(tool_name="project_lookup"),
        )

        self.assertEqual(set(schema["properties"]), {"item"})
        self.assertNotIn("context", schema["properties"])
        self.assertEqual(result, "project_lookup:apollo")

    async def test_decorator_recognizes_annotated_context_with_a_custom_parameter_name(self) -> None:
        @tool
        def current_tool(ctx: ToolExecutionContext) -> str:
            return ctx.tool_name

        parsed = create_schema_adapter(current_tool.schema).validate_python({})
        result = await LocalToolRuntime().execute(
            current_tool,
            parsed,
            ToolExecutionContext(tool_name="current_tool"),
        )

        self.assertEqual(result, "current_tool")

    async def test_input_guardrail_blocks_every_runtime_before_side_effect(self) -> None:
        async def block_input(request) -> ToolGuardrailResult:
            self.assertEqual(request.input, {"item": "apollo"})
            self.assertEqual(request.context.deps, {"tenant": "bank"})
            return ToolGuardrailResult(tripwire_triggered=True, reason="Input denied.")

        definitions = {
            "local": ToolDefinition(
                name="lookup",
                description=None,
                schema={"type": "object"},
                execute=lambda input: input,
                source="local",
                input_guardrails=[block_input],
            ),
            "remote": ToolDefinition(
                name="lookup",
                description=None,
                schema={"type": "object"},
                source="remote",
                requires_approval=False,
                remote_config=RemoteHTTPToolConfig(url="https://example.com/tool"),
                input_guardrails=[block_input],
            ),
            "mcp": ToolDefinition(
                name="lookup",
                description=None,
                schema={"type": "object"},
                source="mcp",
                requires_approval=False,
                mcp_config=MCPToolConfig(
                    server=MCPServerConfig(transport="stdio", command="unused"),
                    tool_name="lookup",
                ),
                input_guardrails=[block_input],
            ),
        }

        for source, definition in definitions.items():
            with self.subTest(source=source):
                runtime = RecordingToolRuntime()
                registry = ToolRegistry({"lookup": definition}, runtimes={source: runtime})
                result = await run_agent(
                    agent=Agent(name="assistant", model=ToolCallingModel(), tools=registry),
                    prompt="lookup",
                    deps={"tenant": "bank"},
                    max_steps=2,
                )

                self.assertEqual(runtime.calls, 0)
                self.assertTrue(result.tool_results[0].is_error)
                self.assertIn("Input denied.", result.tool_results[0].error.message)
                guardrail_events = [
                    event
                    for event in result.trace.events
                    if isinstance(event, AgentGuardrailEvent) and event.metadata.get("scope") == "tool"
                ]
                self.assertEqual(len(guardrail_events), 1)
                self.assertEqual(guardrail_events[0].metadata["tool_name"], "lookup")
                self.assertNotIn("deps", guardrail_events[0].metadata)

    async def test_output_guardrail_can_replace_sensitive_output(self) -> None:
        async def redact_output(request) -> ToolGuardrailResult:
            self.assertEqual(request.output["secret"], "raw-secret")
            return ToolGuardrailResult(
                replacement={"secret": "[REDACTED]", "safe": True},
                replace=True,
                metadata={"policy": "redact-secrets"},
            )

        runtime = RecordingToolRuntime()
        definition = ToolDefinition(
            name="lookup",
            description=None,
            schema={"type": "object"},
            execute=lambda input: input,
            output_guardrails=[redact_output],
        )
        result = await run_agent(
            agent=Agent(
                name="assistant",
                model=ToolCallingModel(),
                tools=ToolRegistry({"lookup": definition}, runtimes={"local": runtime}),
            ),
            prompt="lookup",
            max_steps=2,
        )

        self.assertEqual(runtime.calls, 1)
        self.assertEqual(result.tool_results[0].output, {"secret": "[REDACTED]", "safe": True})
        self.assertNotIn("raw-secret", repr(result.tool_results))

    async def test_input_replacement_is_revalidated_and_approved_as_executed(self) -> None:
        approved_inputs: list[object] = []

        def canonicalize_input(request) -> ToolGuardrailResult:
            return ToolGuardrailResult(replacement={"item": "canonical-apollo"}, replace=True)

        def approve(request) -> bool:
            approved_inputs.append(request.tool_input)
            return True

        runtime = RecordingToolRuntime(output={"ok": True})
        definition = ToolDefinition(
            name="lookup",
            description=None,
            schema={"type": "object"},
            execute=lambda input: input,
            requires_approval=True,
            input_guardrails=[canonicalize_input],
        )
        await run_agent(
            agent=Agent(
                name="assistant",
                model=ToolCallingModel(),
                tools=ToolRegistry({"lookup": definition}, runtimes={"local": runtime}),
                approval_policy=approve,
            ),
            prompt="lookup",
            max_steps=2,
        )

        self.assertEqual(approved_inputs, [{"item": "canonical-apollo"}])
        self.assertEqual(runtime.last_input, {"item": "canonical-apollo"})

    async def test_guardrail_evaluation_error_fails_closed_without_error_detail(self) -> None:
        async def broken_guardrail(request):
            raise RuntimeError("private-policy-dependency-secret")

        runtime = RecordingToolRuntime()
        definition = ToolDefinition(
            name="lookup",
            description=None,
            schema={"type": "object"},
            execute=lambda input: input,
            input_guardrails=[broken_guardrail],
        )
        result = await run_agent(
            agent=Agent(
                name="assistant",
                model=ToolCallingModel(),
                tools=ToolRegistry({"lookup": definition}, runtimes={"local": runtime}),
            ),
            prompt="lookup",
            max_steps=2,
        )

        self.assertEqual(runtime.calls, 0)
        self.assertIn("Guardrail evaluation failed.", result.tool_results[0].error.message)
        self.assertNotIn("private-policy-dependency-secret", result.tool_results[0].error.message)

    def test_tool_execution_context_cancellation_helpers_are_duck_typed_and_hidden(self) -> None:
        class Token:
            cancelled = True

        context = ToolExecutionContext(tool_name="lookup", cancellation_token=Token())

        self.assertTrue(context.cancellation_requested)
        self.assertNotIn("Token", repr(context))
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            context.raise_if_cancelled()
