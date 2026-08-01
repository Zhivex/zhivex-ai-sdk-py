from __future__ import annotations

import unittest
from importlib.util import find_spec

from zhivex_ai import Agent, create_mock_language_model, tool
from zhivex_ai.messages import create_text_message
from zhivex_ai.protocols import (
    A2AAgentExecutor,
    A2A_PROTOCOL_VERSION,
    create_a2a_agent_card,
    create_a2a_app,
    stream_agent_ag_ui,
    to_ag_ui_sse_response,
)
from zhivex_ai.types import GenerateResult, StreamFinishEvent, StreamTextDeltaEvent, StreamToolCallEvent, ToolCall


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
        self.assertEqual(events[1]["statusUpdate"]["status"]["state"], "TASK_STATE_WORKING")
        self.assertIn("artifactUpdate", events[2])
        self.assertEqual(events[-1]["statusUpdate"]["status"]["state"], "TASK_STATE_COMPLETED")


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
                    [StreamToolCallEvent(tool_call=call), StreamFinishEvent(finish_reason="tool-calls")],
                    [StreamTextDeltaEvent(text_delta="done"), StreamFinishEvent(finish_reason="stop")],
                ]
            ),
            tools={"lookup": tool(name="lookup", schema=dict, execute=lambda value: {"found": value["id"]})},
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


@unittest.skipUnless(find_spec("a2a") and find_spec("fastapi"), "A2A optional extra is not installed")
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
        self.assertEqual(response.json()["task"]["status"]["state"], "TASK_STATE_COMPLETED")
