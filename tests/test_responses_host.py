from __future__ import annotations

import json
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from zhivex_ai import Agent, ValidationError, create_mock_language_model
from zhivex_ai.messages import create_text_message
from zhivex_ai.protocols import HostedAgentRunOptions, ProtocolLimits
from zhivex_ai.responses_host import (
    InMemoryResponsesEventStore,
    ResponsesAgentHost,
    create_responses_app,
)
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
        events = [
            event
            async for event in host.stream(
                {"model": "support", "input": "help", "stream": True}
            )
        ]

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
        self.assertEqual(
            events[-1]["response"]["output"][0]["content"][0]["text"], "done"
        )

    async def test_unknown_alias_is_rejected(self) -> None:
        host = ResponsesAgentHost({})
        with self.assertRaisesRegex(KeyError, "missing"):
            await host.create({"model": "missing", "input": "help"})

    async def test_rejects_unsupported_request_fields_and_non_boolean_stream(
        self,
    ) -> None:
        host = ResponsesAgentHost({"support": _agent("done")})

        for field, value in (
            ("store", True),
            ("background", True),
            ("tools", []),
            ("previous_response_id", "resp_previous"),
            ("max_output_tokens", 10),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValidationError, field):
                    await host.create(
                        {"model": "support", "input": "help", field: value}
                    )

        with self.assertRaisesRegex(ValidationError, "stream must be a boolean"):
            await host.create({"model": "support", "input": "help", "stream": "true"})

    async def test_applies_alias_message_part_and_text_limits(self) -> None:
        limits = ProtocolLimits(
            max_alias_chars=5,
            max_messages=1,
            max_parts_per_message=1,
            max_text_chars=4,
        )
        host = ResponsesAgentHost({"short": _agent("done")}, limits=limits)

        with self.assertRaisesRegex(ValidationError, "model alias"):
            await host.create({"model": "longer", "input": "help"})
        with self.assertRaisesRegex(ValidationError, "message limit"):
            await host.create(
                {
                    "model": "short",
                    "input": [
                        {"role": "user", "content": "a"},
                        {"role": "user", "content": "b"},
                    ],
                }
            )
        with self.assertRaisesRegex(ValidationError, "part limit"):
            await host.create(
                {
                    "model": "short",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "a"},
                                {"type": "input_text", "text": "b"},
                            ],
                        }
                    ],
                }
            )
        with self.assertRaisesRegex(ValidationError, "input text"):
            await host.create({"model": "short", "input": "12345"})

    async def test_propagates_trusted_run_options_and_emits_sanitized_events(
        self,
    ) -> None:
        deps = object()
        runtime = object()
        session = object()
        observed: list[dict[str, object]] = []

        async def resolve_options(invocation):
            self.assertEqual(invocation.protocol, "responses")
            return HostedAgentRunOptions(
                session=session,  # type: ignore[arg-type]
                deps=deps,
                idempotency_key="tenant:key",
                runtime=runtime,  # type: ignore[arg-type]
            )

        result = SimpleNamespace(text="done", usage=None, run_id="run_internal")
        with patch(
            "zhivex_ai.responses_host.run_agent", new=AsyncMock(return_value=result)
        ) as mocked:
            host = ResponsesAgentHost(
                {"support": _agent("unused")},
                run_options_resolver=resolve_options,
                on_protocol_event=lambda event: observed.append(dict(event)),
            )
            response = await host.create({"model": "support", "input": "help"})

        self.assertEqual(response["status"], "completed")
        self.assertIs(mocked.await_args.kwargs["session"], session)
        self.assertIs(mocked.await_args.kwargs["deps"], deps)
        self.assertIs(mocked.await_args.kwargs["runtime"], runtime)
        self.assertEqual(mocked.await_args.kwargs["idempotency_key"], "tenant:key")
        self.assertEqual(
            [event["status"] for event in observed], ["started", "completed"]
        )
        self.assertEqual(observed[-1]["internal_run_id"], "run_internal")
        self.assertNotIn("payload", observed[-1])

    async def test_in_memory_event_store_gets_and_replays_after_sequence(self) -> None:
        store = InMemoryResponsesEventStore()
        host = ResponsesAgentHost({"support": _agent("done")}, event_store=store)
        events = [
            event
            async for event in host.stream(
                {"model": "support", "input": "help", "stream": True}
            )
        ]
        response_id = events[0]["response"]["id"]

        record = await host.get(response_id)
        replayed = await host.replay(response_id, after_sequence=4)

        self.assertIsNotNone(record)
        self.assertEqual(record.status, "completed")
        self.assertEqual(record.response["output"][0]["content"][0]["text"], "done")
        self.assertEqual(
            [event["sequence_number"] for event in replayed], list(range(5, 9))
        )

    async def test_stream_errors_are_sanitized_by_default(self) -> None:
        failing = Agent(
            name="assistant",
            model=create_mock_language_model(responses=[]),
        )
        host = ResponsesAgentHost({"support": failing})

        events = [
            event
            async for event in host.stream(
                {"model": "support", "input": "help", "stream": True}
            )
        ]

        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(
            events[-1]["response"]["error"]["message"], "Agent execution failed."
        )
        self.assertNotIn("no responses left", str(events[-1]))

    def test_fastapi_dependency_is_optional(self) -> None:
        try:
            import fastapi  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(RuntimeError, r"\[api\]"):
                create_responses_app(agents={"support": _agent("done")})


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "api extra is not installed")
class ResponsesFastAPITests(unittest.TestCase):
    def test_stream_headers_ids_get_and_last_event_id_replay(self) -> None:
        from fastapi.testclient import TestClient

        store = InMemoryResponsesEventStore()
        app = create_responses_app(
            agents={"support": _agent("done")},
            event_store=store,
        )
        with TestClient(app) as client:
            streamed = client.post(
                "/v1/responses",
                json={"model": "support", "input": "help", "stream": True},
            )
            first_data = next(
                line.removeprefix("data: ")
                for line in streamed.text.splitlines()
                if line.startswith("data: ")
            )
            response_id = json.loads(first_data)["response"]["id"]
            retrieved = client.get(f"/v1/responses/{response_id}")
            replayed = client.get(
                f"/v1/responses/{response_id}/events",
                headers={"Last-Event-ID": "4"},
            )

        self.assertEqual(streamed.status_code, 200)
        self.assertEqual(streamed.headers["cache-control"], "no-cache, no-transform")
        self.assertEqual(streamed.headers["x-accel-buffering"], "no")
        self.assertIn("id: 0\n", streamed.text)
        self.assertEqual(retrieved.json()["status"], "completed")
        self.assertNotIn("id: 4\n", replayed.text)
        self.assertIn("id: 5\n", replayed.text)

    def test_authorization_runs_before_agent_resolution(self) -> None:
        from fastapi.testclient import TestClient

        resolver = Mock(side_effect=AssertionError("must not resolve"))
        app = create_responses_app(
            agents=resolver,
            authorize=lambda request: False,
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "support", "input": "help"},
            )

        self.assertEqual(response.status_code, 401)
        resolver.assert_not_called()

    def test_stream_unknown_alias_returns_json_404_before_sse_starts(self) -> None:
        from fastapi.testclient import TestClient

        app = create_responses_app(agents={"support": _agent("done")})
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "missing", "input": "help", "stream": True},
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(
            response.json(),
            {
                "error": {
                    "type": "invalid_request_error",
                    "message": 'Unknown model alias "missing".',
                }
            },
        )

    def test_stream_validation_returns_json_400_before_sse_starts(self) -> None:
        from fastapi.testclient import TestClient

        app = create_responses_app(agents={"support": _agent("done")})
        cases = (
            (
                {"model": "support", "input": 123, "stream": True},
                "Responses input must be a non-empty string or item list.",
            ),
            (
                {
                    "model": "support",
                    "input": "help",
                    "stream": True,
                    "tools": [],
                },
                "Unsupported Responses request field(s): tools.",
            ),
        )

        with TestClient(app) as client:
            for payload, message in cases:
                with self.subTest(payload=payload):
                    response = client.post("/v1/responses", json=payload)
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(
                        response.headers["content-type"], "application/json"
                    )
                    self.assertEqual(
                        response.json(),
                        {
                            "error": {
                                "type": "invalid_request_error",
                                "message": message,
                            }
                        },
                    )

    def test_non_stream_failures_do_not_expose_internal_error(self) -> None:
        from fastapi.testclient import TestClient

        failing = Agent(
            name="assistant",
            model=create_mock_language_model(responses=[]),
        )
        app = create_responses_app(agents={"support": failing})
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "support", "input": "help"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["message"], "Agent execution failed.")
        self.assertNotIn("no responses left", response.text)

    def test_non_stream_error_mapper_runs_once(self) -> None:
        from fastapi.testclient import TestClient

        mapper = Mock(return_value="Reviewed failure.")
        app = create_responses_app(
            agents={
                "support": Agent(
                    name="assistant",
                    model=create_mock_language_model(responses=[]),
                )
            },
            error_mapper=mapper,
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "support", "input": "help"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["message"], "Reviewed failure.")
        mapper.assert_called_once()
