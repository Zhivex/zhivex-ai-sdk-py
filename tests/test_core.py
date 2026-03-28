from __future__ import annotations

from collections.abc import AsyncIterable
import sys
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
from zhivex_ai.types import GenerateResult, ModelGenerateInput, StreamFinishEvent, StreamTextDeltaEvent, ToolCall, ToolCallPart


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
                ]
            )
        return GenerateResult(messages=[create_text_message("assistant", '{"city":"Madrid","forecast":"sunny"}')])

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        async def generator() -> AsyncIterable[object]:
            yield StreamTextDeltaEvent(text_delta="hello")
            yield StreamTextDeltaEvent(text_delta=" world")
            yield StreamFinishEvent(finish_reason="stop")

        return generator()


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
