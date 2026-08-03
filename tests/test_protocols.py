from __future__ import annotations

import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import patch

from zhivex_ai import Agent, ValidationError, create_mock_language_model, tool
from zhivex_ai.messages import create_text_message
from zhivex_ai.protocols import (
    A2AAgentExecutor,
    A2A_PROTOCOL_VERSION,
    HostedAgentRunOptions,
    ProtocolLimits,
    create_a2a_agent_card,
    create_a2a_app,
    stream_agent_ag_ui,
    to_ag_ui_sse_response,
)
from zhivex_ai.types import (
    GenerateResult,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    ToolCall,
)


def _agent(text: str) -> Agent:
    return Agent(
        name="support",
        model=create_mock_language_model(
            responses=[
                GenerateResult(
                    text=text,
                    message=create_text_message("assistant", text),
                    finish_reason="stop",
                )
            ]
        ),
    )


class A2AProtocolTests(unittest.IsolatedAsyncioTestCase):
    def test_agent_card_uses_a2a_v1_supported_interfaces(self) -> None:
        card = create_a2a_agent_card(
            _agent("ok"),
            url="https://agents.example.com/support/",
            version="1.2.3",
        ).to_dict()

        self.assertEqual(
            card["supportedInterfaces"],
            [
                {
                    "url": "https://agents.example.com/support",
                    "protocolBinding": "HTTP+JSON",
                    "protocolVersion": A2A_PROTOCOL_VERSION,
                }
            ],
        )
        self.assertEqual(card["skills"][0]["id"], "support")

    async def test_send_message_returns_output_as_task_artifact(self) -> None:
        executor = A2AAgentExecutor(_agent("resolved"))
        task = await executor.send_message(
            {
                "message": {
                    "messageId": "msg-1",
                    "role": "user",
                    "parts": [{"text": "help"}],
                }
            }
        )

        self.assertEqual(task["status"]["state"], "TASK_STATE_COMPLETED")
        self.assertEqual(task["artifacts"][0]["parts"], [{"text": "resolved"}])
        self.assertEqual(executor.get_task(task["id"]), task)

    async def test_stream_message_emits_v1_discriminated_updates(self) -> None:
        executor = A2AAgentExecutor(_agent("streamed"))
        events = [
            event
            async for event in executor.stream_message(
                {
                    "message": {
                        "messageId": "msg-1",
                        "role": "user",
                        "parts": [{"text": "help"}],
                    }
                }
            )
        ]

        self.assertEqual(list(events[0]), ["task"])
        self.assertEqual(events[0]["task"]["status"]["state"], "TASK_STATE_SUBMITTED")
        self.assertEqual(
            events[1]["statusUpdate"]["status"]["state"], "TASK_STATE_WORKING"
        )
        self.assertIn("artifactUpdate", events[2])
        self.assertEqual(
            events[-1]["statusUpdate"]["status"]["state"], "TASK_STATE_COMPLETED"
        )

    async def test_rejects_excessive_parts_and_sanitizes_failures(self) -> None:
        limited = A2AAgentExecutor(
            _agent("unused"),
            limits=ProtocolLimits(max_parts_per_message=1),
        )
        with self.assertRaisesRegex(ValidationError, "part limit"):
            await limited.send_message(
                {
                    "message": {
                        "messageId": "msg-1",
                        "role": "user",
                        "parts": [{"text": "one"}, {"text": "two"}],
                    }
                }
            )

        failing = A2AAgentExecutor(
            Agent(name="support", model=create_mock_language_model(responses=[])),
        )
        task = await failing.send_message(
            {
                "message": {
                    "messageId": "msg-2",
                    "role": "user",
                    "parts": [{"text": "help"}],
                }
            }
        )

        self.assertEqual(task["status"]["state"], "TASK_STATE_FAILED")
        self.assertEqual(
            task["status"]["message"]["parts"][0]["text"], "Agent execution failed."
        )
        self.assertNotIn("no responses left", str(task))


class AGUIProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_translates_agent_stream_to_ag_ui_lifecycle(self) -> None:
        events = [
            event
            async for event in stream_agent_ag_ui(
                agent=_agent("hello"),
                prompt="hi",
                thread_id="thread-1",
                run_id="run-1",
            )
        ]

        self.assertEqual(
            [event["type"] for event in events],
            [
                "RUN_STARTED",
                "TEXT_MESSAGE_START",
                "TEXT_MESSAGE_CONTENT",
                "TEXT_MESSAGE_END",
                "RUN_FINISHED",
            ],
        )
        self.assertEqual(events[0]["threadId"], "thread-1")
        self.assertEqual(events[2]["delta"], "hello")

    async def test_translates_tool_calls_and_results(self) -> None:
        call = ToolCall(id="call-1", name="lookup", input={"id": 7})
        agent = Agent(
            name="support",
            model=create_mock_language_model(
                stream_events=[
                    [
                        StreamToolCallEvent(tool_call=call),
                        StreamFinishEvent(finish_reason="tool-calls"),
                    ],
                    [
                        StreamTextDeltaEvent(text_delta="done"),
                        StreamFinishEvent(finish_reason="stop"),
                    ],
                ]
            ),
            tools={
                "lookup": tool(
                    name="lookup",
                    schema=dict,
                    execute=lambda value: {"found": value["id"]},
                )
            },
        )

        events = [
            event
            async for event in stream_agent_ag_ui(
                agent=agent,
                prompt="find",
                thread_id="thread-1",
                run_id="run-1",
            )
        ]
        event_types = [event["type"] for event in events]

        self.assertIn("TOOL_CALL_START", event_types)
        self.assertIn("TOOL_CALL_ARGS", event_types)
        self.assertIn("TOOL_CALL_END", event_types)
        self.assertIn("TOOL_CALL_RESULT", event_types)
        result = next(event for event in events if event["type"] == "TOOL_CALL_RESULT")
        self.assertEqual(result["toolCallId"], "call-1")
        self.assertEqual(result["content"], '{"found": 7}')

    async def test_propagates_run_options_and_emits_sanitized_observer_events(
        self,
    ) -> None:
        captured: dict[str, object] = {}
        observed: list[dict[str, object]] = []
        deps = object()
        runtime = object()
        session = object()

        class FakeStream:
            async def event_stream(self):
                from zhivex_ai.agent import AgentRunStartEvent, AgentTextDeltaEvent

                yield AgentRunStartEvent(
                    run_id="run-internal", session_id="session", agent_name="support"
                )
                yield AgentTextDeltaEvent(text_delta="hello")

            async def collect(self):
                return SimpleNamespace(text="hello", run_id="run-internal")

        def fake_stream_agent(**kwargs):
            captured.update(kwargs)
            return FakeStream()

        with patch("zhivex_ai.protocols.stream_agent", side_effect=fake_stream_agent):
            events = [
                event
                async for event in stream_agent_ag_ui(
                    agent=_agent("unused"),
                    prompt="hi",
                    thread_id="thread-1",
                    run_id="run-1",
                    run_options=HostedAgentRunOptions(
                        session=session,  # type: ignore[arg-type]
                        deps=deps,
                        idempotency_key="tenant:key",
                        runtime=runtime,  # type: ignore[arg-type]
                    ),
                    on_protocol_event=lambda event: observed.append(dict(event)),
                )
            ]

        self.assertEqual(events[-1]["type"], "RUN_FINISHED")
        self.assertIs(captured["session"], session)
        self.assertIs(captured["deps"], deps)
        self.assertIs(captured["runtime"], runtime)
        self.assertEqual(captured["idempotency_key"], "tenant:key")
        self.assertEqual(
            [event["status"] for event in observed], ["started", "running", "completed"]
        )
        self.assertEqual(observed[-1]["internal_run_id"], "run-internal")
        self.assertNotIn("payload", observed[-1])

    async def test_maps_ag_ui_errors_without_exposing_internal_details(self) -> None:
        agent = Agent(name="support", model=create_mock_language_model(responses=[]))
        events = [
            event
            async for event in stream_agent_ag_ui(
                agent=agent,
                prompt="hi",
                thread_id="thread-1",
                error_mapper=lambda error, invocation: "Public failure.",
            )
        ]

        self.assertEqual(events[-1]["type"], "RUN_ERROR")
        self.assertEqual(events[-1]["message"], "Public failure.")
        self.assertEqual(events[-1]["code"], "agent_execution_error")
        self.assertNotIn("no responses left", str(events[-1]))

    @unittest.skipUnless(find_spec("ag_ui"), "ag-ui optional extra is not installed")
    async def test_official_encoder_validates_and_encodes_events(self) -> None:
        response = to_ag_ui_sse_response(
            stream_agent_ag_ui(
                agent=_agent("hello"),
                prompt="hi",
                thread_id="thread-1",
                run_id="run-1",
            )
        )
        frames = [frame.decode("utf-8") async for frame in response.body]

        self.assertEqual(response.headers["content-type"], "text/event-stream")
        self.assertIn('"type":"RUN_STARTED"', frames[0])
        self.assertIn('"type":"RUN_FINISHED"', frames[-1])


@unittest.skipUnless(
    find_spec("a2a") and find_spec("fastapi"), "A2A optional extra is not installed"
)
class OfficialA2AServerTests(unittest.TestCase):
    def test_official_rest_binding_completes_task(self) -> None:
        from fastapi.testclient import TestClient

        agent = _agent("resolved")
        app = create_a2a_app(
            executor=A2AAgentExecutor(agent),
            card=create_a2a_agent_card(
                agent,
                url="http://testserver/a2a",
                version="0.16.0",
            ),
        )
        with TestClient(app) as client:
            card = client.get("/.well-known/agent-card.json")
            response = client.post(
                "/a2a/message:send",
                headers={"A2A-Version": A2A_PROTOCOL_VERSION},
                json={
                    "message": {
                        "messageId": "msg-1",
                        "role": "ROLE_USER",
                        "parts": [{"text": "help"}],
                    }
                },
            )

        self.assertEqual(card.status_code, 200)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["task"]["status"]["state"], "TASK_STATE_COMPLETED"
        )

    def test_authorizes_every_route_except_exact_public_agent_card(self) -> None:
        from fastapi.testclient import TestClient

        agent = _agent("resolved")
        app = create_a2a_app(
            executor=A2AAgentExecutor(agent),
            card=create_a2a_agent_card(
                agent,
                url="http://testserver/a2a",
                version="0.16.0",
            ),
            authorize=lambda request: False,
        )
        with TestClient(app) as client:
            card = client.get("/.well-known/agent-card.json")
            task = client.get(
                "/a2a/tasks/task-missing",
                headers={"A2A-Version": A2A_PROTOCOL_VERSION},
            )
            near_card = client.get("/.well-known/agent-card.json/extra")

        self.assertEqual(card.status_code, 200)
        self.assertEqual(task.status_code, 401)
        self.assertEqual(near_card.status_code, 401)

    def test_measures_actual_body_even_when_content_length_is_smaller(self) -> None:
        from fastapi.testclient import TestClient

        agent = _agent("resolved")
        app = create_a2a_app(
            executor=A2AAgentExecutor(agent),
            card=create_a2a_agent_card(
                agent,
                url="http://testserver/a2a",
                version="0.16.0",
            ),
            max_request_bytes=32,
        )
        with TestClient(app) as client:
            response = client.post(
                "/a2a/message:send",
                headers={
                    "A2A-Version": A2A_PROTOCOL_VERSION,
                    "Content-Length": "1",
                    "Content-Type": "application/json",
                },
                content=b"{" + (b'"padding":"' + b"x" * 128 + b'"}'),
            )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json(), {"error": "request_too_large"})

    def test_accepts_official_store_context_builder_and_queue_manager(self) -> None:
        from a2a.server.agent_execution import SimpleRequestContextBuilder
        from a2a.server.events import InMemoryQueueManager
        from a2a.server.tasks import InMemoryTaskStore
        from fastapi.testclient import TestClient

        saved: list[str] = []

        class SpyTaskStore(InMemoryTaskStore):
            async def save(self, task, context):
                saved.append(task.id)
                await super().save(task, context)

        agent = _agent("resolved")
        task_store = SpyTaskStore()
        app = create_a2a_app(
            executor=A2AAgentExecutor(agent),
            card=create_a2a_agent_card(
                agent,
                url="http://testserver/a2a",
                version="0.16.0",
            ),
            task_store=task_store,
            request_context_builder=SimpleRequestContextBuilder(task_store=task_store),
            queue_manager=InMemoryQueueManager(),
        )
        with TestClient(app) as client:
            response = client.post(
                "/a2a/message:send",
                headers={"A2A-Version": A2A_PROTOCOL_VERSION},
                json={
                    "message": {
                        "messageId": "msg-1",
                        "role": "ROLE_USER",
                        "parts": [{"text": "help"}],
                    }
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(saved)
