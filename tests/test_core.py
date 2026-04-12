from __future__ import annotations

from collections.abc import AsyncIterable
import sys
import asyncio
from dataclasses import replace
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    FilePart,
    ModelCapabilities,
    ModelMessage,
    ReasoningConfig,
    ToolChoiceName,
    ToolExecutionOptions,
    ValidationError,
    create_text_message,
    generate_grounded_text,
    generate_object,
    generate_text,
    stream_text,
    tool,
)
from zhivex_ai.schema import create_schema_adapter
from zhivex_ai.types import (
    GenerateResult,
    ModelGenerateInput,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    GroundedGenerateResult,
    GroundingSource,
    GroundedModelGenerateInput,
)


class FakeLanguageModel:
    provider = "test"
    model_id = "model"
    capabilities = ModelCapabilities(
        streaming=True,
        tools=True,
        structured_output=True,
        json_mode=True,
        tool_choice=True,
        parallel_tool_calls=False,
        vision=True,
        files=False,
        audio_input=False,
        audio_output=False,
        embeddings=False,
        reasoning=True,
        web_search=False,
    )

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        self.calls += 1
        if self.calls == 1 and input.tools:
            return GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[ToolCallPart(tool_call=ToolCall(id="1", name="weather", input={"city": "Madrid"}))],
                    )
                ],
                usage=TokenUsage(input_tokens=10, output_tokens=2, total_tokens=12),
            )
        return GenerateResult(
            messages=[create_text_message("assistant", '{"city":"Madrid","forecast":"sunny"}')],
            usage=TokenUsage(input_tokens=8, output_tokens=4, total_tokens=12),
        )

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        async def generator() -> AsyncIterable[object]:
            yield StreamTextDeltaEvent(text_delta="hello")
            yield StreamTextDeltaEvent(text_delta=" world")
            yield StreamFinishEvent(
                finish_reason="stop",
                usage=TokenUsage(input_tokens=3, output_tokens=2, total_tokens=5),
            )

        return generator()


class FakeStreamingToolModel(FakeLanguageModel):
    def __init__(self) -> None:
        super().__init__()
        self.stream_calls = 0

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        self.stream_calls += 1

        async def generator() -> AsyncIterable[object]:
            if self.stream_calls == 1 and input.tools:
                yield StreamToolCallEvent(tool_call=ToolCall(id="1", name="weather", input={"city": "Madrid"}))
                yield StreamFinishEvent(
                    finish_reason="tool-calls",
                    usage=TokenUsage(input_tokens=10, output_tokens=2, total_tokens=12),
                )
                return
            yield StreamTextDeltaEvent(text_delta='{"city":"Madrid","forecast":"sunny"}')
            yield StreamFinishEvent(
                finish_reason="stop",
                usage=TokenUsage(input_tokens=8, output_tokens=4, total_tokens=12),
            )

        return generator()


class FakeParallelToolModel(FakeLanguageModel):
    capabilities = ModelCapabilities(
        streaming=True,
        tools=True,
        structured_output=True,
        json_mode=True,
        tool_choice=True,
        parallel_tool_calls=True,
        vision=True,
        files=False,
        audio_input=False,
        audio_output=False,
        embeddings=False,
        reasoning=True,
        web_search=False,
    )

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        self.calls += 1
        if self.calls == 1 and input.tools:
            return GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[
                            ToolCallPart(tool_call=ToolCall(id="1", name="weather", input={"city": "Madrid"})),
                            ToolCallPart(tool_call=ToolCall(id="2", name="timezone", input={"city": "Madrid"})),
                        ],
                    )
                ]
            )
        return GenerateResult(messages=[create_text_message("assistant", "done")])


class Forecast(BaseModel):
    city: str
    forecast: str


class FakeGroundedModel:
    provider = "test"
    model_id = "grounded-model"
    capabilities = ModelCapabilities(
        streaming=False,
        tools=False,
        structured_output=False,
        json_mode=False,
        tool_choice=False,
        parallel_tool_calls=False,
        vision=False,
        files=False,
        audio_input=False,
        audio_output=False,
        embeddings=False,
        reasoning=True,
        web_search=True,
    )

    async def generate(self, input: GroundedModelGenerateInput) -> GroundedGenerateResult:
        return GroundedGenerateResult(
            text="grounded",
            usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            sources=[GroundingSource(url="https://example.com", title="Example")],
        )


class FakeFileLanguageModel(FakeLanguageModel):
    capabilities = replace(FakeLanguageModel.capabilities, files=True)

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        return GenerateResult(messages=[create_text_message("assistant", "ok")], text="ok")


class CoreTests(IsolatedAsyncioTestCase):
    def test_schema_adapter_supports_raw_json_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        }
        adapter = create_schema_adapter(schema)
        self.assertEqual(adapter.json_schema(), schema)
        self.assertEqual(adapter.validate_python({"city": "Madrid"}), {"city": "Madrid"})

    async def test_generate_text_executes_tools(self) -> None:
        model = FakeLanguageModel()
        result = await generate_text(
            model=model,
            prompt="Weather?",
            max_steps=2,
            tools={
                "weather": tool(
                    name="weather",
                    schema=dict[str, str],
                    execute=lambda input: {"city": input["city"], "forecast": "sunny"},
                )
            },
            reasoning=ReasoningConfig(effort="medium"),
        )
        self.assertEqual(result.text, '{"city":"Madrid","forecast":"sunny"}')
        self.assertEqual(result.tool_results[0].tool_name, "weather")
        self.assertEqual(result.usage.input_tokens, 18)
        self.assertEqual(result.usage.output_tokens, 6)
        self.assertEqual(result.usage.total_tokens, 24)

    async def test_generate_text_returns_only_the_latest_assistant_reply(self) -> None:
        class FinalReplyModel(FakeLanguageModel):
            async def generate(self, input: ModelGenerateInput) -> GenerateResult:
                return GenerateResult(text="new reply")

        result = await generate_text(
            model=FinalReplyModel(),
            messages=[
                create_text_message("user", "hi"),
                create_text_message("assistant", "old reply"),
                create_text_message("user", "what now?"),
            ],
        )

        self.assertEqual(result.text, "new reply")

    async def test_generate_object_parses_schema(self) -> None:
        model = FakeLanguageModel()
        result = await generate_object(model=model, prompt="Return JSON", schema=Forecast)
        self.assertEqual(result.object.city, "Madrid")
        self.assertEqual(result.object.forecast, "sunny")

    async def test_generate_object_ignores_prior_assistant_history_when_parsing(self) -> None:
        class FinalJsonModel(FakeLanguageModel):
            async def generate(self, input: ModelGenerateInput) -> GenerateResult:
                return GenerateResult(text='{"city":"Madrid","forecast":"sunny"}')

        result = await generate_object(
            model=FinalJsonModel(),
            messages=[
                create_text_message("user", "hi"),
                create_text_message("assistant", '{"city":"Paris","forecast":"rainy"}'),
                create_text_message("user", "return Madrid weather"),
            ],
            schema=Forecast,
        )

        self.assertEqual(result.text, '{"city":"Madrid","forecast":"sunny"}')
        self.assertEqual(result.object.city, "Madrid")
        self.assertEqual(result.object.forecast, "sunny")

    async def test_stream_text_collects_text(self) -> None:
        model = FakeLanguageModel()
        result = stream_text(model=model, prompt="hello")
        final = await result.collect()
        self.assertEqual(final.text, "hello world")
        self.assertEqual(final.usage.total_tokens, 5)

    async def test_generate_text_accepts_pdf_file_part_with_single_source(self) -> None:
        result = await generate_text(
            model=FakeFileLanguageModel(),
            messages=[
                ModelMessage(
                    role="user",
                    parts=[FilePart(data="JVBERi0xLjQK", media_type="application/pdf", filename="statement.pdf")],
                )
            ],
        )
        self.assertEqual(result.text, "ok")

    async def test_generate_text_rejects_file_part_with_multiple_sources(self) -> None:
        with self.assertRaises(ValidationError) as context:
            await generate_text(
                model=FakeFileLanguageModel(),
                messages=[
                    ModelMessage(
                        role="user",
                        parts=[FilePart(data="JVBERi0xLjQK", media_type="application/pdf", file_id="file_123")],
                    )
                ],
            )

        self.assertIn("exactly one source", str(context.exception))

    async def test_generate_text_accepts_non_pdf_file_parts(self) -> None:
        result = await generate_text(
            model=FakeFileLanguageModel(),
            messages=[
                ModelMessage(role="user", parts=[FilePart(data="hello", media_type="text/plain", filename="notes.txt")])
            ],
        )

        self.assertEqual(result.text, "ok")

    async def test_generate_text_allows_file_references_without_media_type(self) -> None:
        result = await generate_text(
            model=FakeFileLanguageModel(),
            messages=[ModelMessage(role="user", parts=[FilePart(file_id="file_123", filename="statement.pdf")])],
        )

        self.assertEqual(result.text, "ok")

    async def test_stream_text_aggregates_usage_across_tool_steps(self) -> None:
        model = FakeStreamingToolModel()
        result = stream_text(
            model=model,
            prompt="Weather?",
            max_steps=2,
            tools={
                "weather": tool(
                    name="weather",
                    schema=dict[str, str],
                    execute=lambda input: {"city": input["city"], "forecast": "sunny"},
                )
            },
        )
        final = await result.collect()
        self.assertEqual(final.text, '{"city":"Madrid","forecast":"sunny"}')
        self.assertEqual(final.usage.input_tokens, 18)
        self.assertEqual(final.usage.output_tokens, 6)
        self.assertEqual(final.usage.total_tokens, 24)

    async def test_generate_text_executes_parallel_tool_calls_concurrently(self) -> None:
        model = FakeParallelToolModel()
        state = {"in_flight": 0, "max_in_flight": 0}

        async def execute_tool(name: str, city: str) -> dict[str, str]:
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            try:
                await asyncio.sleep(0.01)
                return {"tool": name, "city": city}
            finally:
                state["in_flight"] -= 1

        result = await generate_text(
            model=model,
            prompt="Weather and timezone?",
            max_steps=2,
            tools={
                "weather": tool(
                    name="weather",
                    schema=dict[str, str],
                    execute=lambda input: execute_tool("weather", input["city"]),
                ),
                "timezone": tool(
                    name="timezone",
                    schema=dict[str, str],
                    execute=lambda input: execute_tool("timezone", input["city"]),
                ),
            },
        )
        self.assertEqual(result.text, "done")
        self.assertEqual(state["max_in_flight"], 2)
        self.assertEqual([tool_result.tool_name for tool_result in result.tool_results], ["weather", "timezone"])

    async def test_generate_text_accepts_named_tool_choice(self) -> None:
        model = FakeLanguageModel()
        await generate_text(
            model=model,
            prompt="Weather?",
            tools={
                "weather": tool(
                    name="weather",
                    schema=dict[str, str],
                    execute=lambda input: {"city": input["city"], "forecast": "sunny"},
                )
            },
            tool_choice=ToolChoiceName(tool_name="weather"),
        )
        self.assertEqual(model.calls, 1)

    async def test_generate_text_stops_on_tool_error_when_requested(self) -> None:
        model = FakeLanguageModel()

        with self.assertRaises(RuntimeError):
            await generate_text(
                model=model,
                prompt="Weather?",
                max_steps=2,
                tools={
                    "weather": tool(
                        name="weather",
                        schema=dict[str, str],
                        execute=lambda input: (_ for _ in ()).throw(RuntimeError("boom")),
                    )
                },
                tool_execution=ToolExecutionOptions(stop_on_error=True),
            )

    async def test_generate_grounded_text_returns_sources(self) -> None:
        result = await generate_grounded_text(model=FakeGroundedModel(), prompt="search")
        self.assertEqual(result.text, "grounded")
        self.assertEqual(result.sources[0].url, "https://example.com")
