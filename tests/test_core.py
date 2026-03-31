from __future__ import annotations

from collections.abc import AsyncIterable
import sys
import asyncio
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    ModelCapabilities,
    ModelMessage,
    ReasoningConfig,
    create_text_message,
    generate_object,
    generate_text,
    stream_text,
    tool,
)
from zhivex_ai.types import (
    GenerateResult,
    ModelGenerateInput,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    TokenUsage,
    ToolCall,
    ToolCallPart,
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


class CoreTests(IsolatedAsyncioTestCase):
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

    async def test_generate_object_parses_schema(self) -> None:
        model = FakeLanguageModel()
        result = await generate_object(model=model, prompt="Return JSON", schema=Forecast)
        self.assertEqual(result.object.city, "Madrid")
        self.assertEqual(result.object.forecast, "sunny")

    async def test_stream_text_collects_text(self) -> None:
        model = FakeLanguageModel()
        result = stream_text(model=model, prompt="hello")
        final = await result.collect()
        self.assertEqual(final.text, "hello world")
        self.assertEqual(final.usage.total_tokens, 5)

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
