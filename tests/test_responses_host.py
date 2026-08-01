from __future__ import annotations

import unittest

from zhivex_ai import Agent, create_mock_language_model
from zhivex_ai.messages import create_text_message
from zhivex_ai.responses_host import ResponsesAgentHost, create_responses_app
from zhivex_ai.types import GenerateResult, TokenUsage


def _agent(text: str) -> Agent:
    return Agent(
        name="assistant",
        model=create_mock_language_model(
            responses=[
                GenerateResult(
                    text=text,
                    message=create_text_message("assistant", text),
                    finish_reason="stop",
                    usage=TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
                )
            ]
        ),
    )


class ResponsesAgentHostTests(unittest.IsolatedAsyncioTestCase):
    async def test_creates_responses_compatible_output(self) -> None:
        host = ResponsesAgentHost({"support": _agent("done")})
        response = await host.create({"model": "support", "input": "help"})

        self.assertEqual(response["object"], "response")
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["model"], "support")
        self.assertEqual(response["output"][0]["content"][0]["text"], "done")
        self.assertEqual(response["usage"]["total_tokens"], 5)

    async def test_streams_official_responses_event_sequence(self) -> None:
        host = ResponsesAgentHost({"support": _agent("done")})
        events = [event async for event in host.stream({"model": "support", "input": "help", "stream": True})]

        self.assertEqual(
            [event["type"] for event in events],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(events[-1]["response"]["output"][0]["content"][0]["text"], "done")

    async def test_unknown_alias_is_rejected(self) -> None:
        host = ResponsesAgentHost({})
        with self.assertRaisesRegex(KeyError, "missing"):
            await host.create({"model": "missing", "input": "help"})

    def test_fastapi_dependency_is_optional(self) -> None:
        try:
            import fastapi  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(RuntimeError, r"\[api\]"):
                create_responses_app(agents={"support": _agent("done")})
