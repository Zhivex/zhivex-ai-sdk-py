from __future__ import annotations

import asyncio
import unittest

from zhivex_ai import (
    Agent,
    EgressDeniedError,
    ModelCatalogEntry,
    ModelResolutionError,
    RunLimits,
    ValidationError,
    clear_registered_model_providers,
    create_egress_policy,
    create_gateway_decide,
    create_latam_sensitive_data_policy,
    create_mock_language_model,
    create_mock_tool,
    create_model_catalog,
    create_model_resolver,
    create_openai_decision_model,
    enforce_egress_policy,
    generate_text,
    has_tool_call,
    openai_compatible,
    register_model_provider,
    resolve_egress_target,
    resolve_language_model,
    step_count_is,
)
from zhivex_ai.decisions import DecisionCapabilities, DecisionInput, DecisionResult, GatewayDecisionTarget
from zhivex_ai.errors import ConfigurationError, ProviderHTTPError
from zhivex_ai.providers.base import ProviderAdapter
from zhivex_ai.types import GenerateResult, ModelMessage, TextPart, ToolCall, ToolCallPart


def _tool_result(name: str, call_id: str) -> GenerateResult:
    return GenerateResult(
        messages=[ModelMessage(role="assistant", parts=[ToolCallPart(tool_call=ToolCall(id=call_id, name=name, input={}))])],
        finish_reason="tool-calls",
    )


def _text_result(text: str = "done") -> GenerateResult:
    return GenerateResult(messages=[ModelMessage(role="assistant", parts=[TextPart(text=text)])], text=text, finish_reason="stop")


class LoopControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_ceiling_stays_one_step(self) -> None:
        model = create_mock_language_model(responses=[_tool_result("read", "1"), _text_result()])
        output = await generate_text(model=model, prompt="read", tools={"read": create_mock_tool("read")})
        self.assertEqual(output.text, "")
        self.assertEqual(len(output.steps), 1)

    async def test_stop_when_uses_twenty_step_ceiling_and_can_stop_earlier(self) -> None:
        model = create_mock_language_model(responses=[_tool_result("read", str(index)) for index in range(5)])
        output = await generate_text(
            model=model,
            prompt="read",
            tools={"read": create_mock_tool("read")},
            stop_when=step_count_is(3),
        )
        self.assertEqual(len(output.steps), 3)

    async def test_unlimited_runs_past_the_default_ceiling(self) -> None:
        model = create_mock_language_model(responses=[*[_tool_result("read", str(index)) for index in range(4)], _text_result("done")])
        output = await generate_text(
            model=model,
            prompt="read",
            tools={"read": create_mock_tool("read")},
            max_steps="unlimited",
        )
        self.assertEqual(output.text, "done")
        self.assertEqual(len(output.steps), 5)

    async def test_invalid_max_steps_is_rejected_before_the_provider_call(self) -> None:
        model = create_mock_language_model(responses=[_text_result()])
        with self.assertRaises(ValidationError):
            await generate_text(model=model, prompt="read", max_steps="infinite")
        self.assertEqual(model.calls, 0)

    async def test_active_tools_and_prepare_step_select_the_visible_tool(self) -> None:
        seen: list[list[str]] = []

        class RecordingModel:
            provider = "test"
            model_id = "model"
            capabilities = create_mock_language_model().capabilities
            calls = 0

            async def generate(self, model_input):
                self.calls += 1
                seen.append(sorted((model_input.tools or {}).keys()))
                return _text_result("ok")

            async def stream(self, model_input):
                raise AssertionError("stream was not requested")

        model = RecordingModel()
        await generate_text(
            model=model,
            prompt="read",
            tools={"read": create_mock_tool("read"), "write": create_mock_tool("write")},
            active_tools=["write"],
        )
        self.assertEqual(seen, [["write"]])

        def prepare(context):
            return {"active_tools": ["read"] if context.step_number == 0 else ["write"]}

        seen.clear()
        model.calls = 0
        await generate_text(
            model=model,
            prompt="read",
            tools={"read": create_mock_tool("read"), "write": create_mock_tool("write")},
            max_steps=2,
            prepare_step=prepare,
            stop_when=has_tool_call("missing"),
        )
        self.assertEqual(seen[0], ["read"])

    async def test_agent_unlimited_respects_a_finite_run_limit(self) -> None:
        from zhivex_ai import run_agent

        model = create_mock_language_model(responses=[_tool_result("read", str(index)) for index in range(12)])
        agent = Agent(name="worker", model=model, tools={"read": create_mock_tool("read")})
        await run_agent(agent=agent, prompt="read", max_steps="unlimited")
        self.assertEqual(model.calls, 8)
        open_agent = Agent(
            name="open",
            model=create_mock_language_model(responses=[*[_tool_result("read", str(index)) for index in range(2)], _text_result("done")]),
            tools={"read": create_mock_tool("read")},
            run_limits=RunLimits(max_steps=None),
        )
        result = await run_agent(agent=open_agent, prompt="read", max_steps="unlimited")
        self.assertEqual(result.text, "done")
        self.assertEqual(open_agent.model.calls, 3)


class ResolverTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_registered_model_providers()

    def test_provider_model_strings_resolve_through_the_catalog(self) -> None:
        catalog = create_model_catalog([ModelCatalogEntry("demo", "chat-small", aliases=("chat",))])
        model = create_mock_language_model(provider="demo", model_id="chat-small")
        adapter = ProviderAdapter(name="demo", language_model_factory=lambda model_id: model)
        resolver = create_model_resolver(
            catalog=catalog,
            adapters={"demo": adapter},
            aliases=[{"alias": "support", "target": "demo/chat"}],
        )
        resolved, metadata = resolver.resolve("support")
        self.assertIs(resolved, model)
        self.assertEqual(metadata.resolved.model_id, "chat-small")
        self.assertEqual(metadata.alias, "support")
        register_model_provider(adapter)
        self.assertIs(resolve_language_model("demo/chat-small"), model)
        with self.assertRaises(ModelResolutionError) as raised:
            resolve_language_model("missing/chat-small")
        self.assertEqual(raised.exception.code, "unknown_provider")

    def test_openai_compatible_factory_uses_chat_completions(self) -> None:
        captured: dict[str, object] = {}

        class Response:
            status_code = 200

            async def json(self):
                return {
                    "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                }

            async def text(self):
                return ""

        async def fetch(url, **kwargs):
            captured["url"] = url
            captured["body"] = kwargs.get("json_body")
            return Response()

        with self.assertRaises(ModelResolutionError) as missing:
            openai_compatible(name="acme", base_url="https://models.example/v1", api_key="")
        self.assertEqual(missing.exception.code, "missing_credentials")
        provider = openai_compatible(name="acme", base_url="https://models.example/v1", api_key="test-key", fetch=fetch, register=True)
        model = provider.language_model("chat")
        result = asyncio.run(model.generate(type("Input", (), {
            "messages": [ModelMessage(role="user", parts=[TextPart(text="hi")])],
            "tools": None,
            "tool_choice": None,
            "temperature": None,
            "max_tokens": None,
            "reasoning": None,
            "provider_options": None,
            "structured_output": None,
            "timeout_ms": None,
            "max_retries": 0,
            "retry_backoff_ms": 0,
        })()))
        self.assertEqual(result.text, "hello")
        self.assertEqual(captured["url"], "https://models.example/v1/chat/completions")
        self.assertEqual(resolve_language_model("acme/chat").model_id, "chat")


class EgressAndRedactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_allowlist_fails_closed(self) -> None:
        policy = create_egress_policy(allow=[])
        with self.assertRaises(EgressDeniedError) as denied:
            await policy.assert_call({"provider": "openai", "model": "gpt-6-luna"})
        self.assertEqual(denied.exception.audit.reason, "allowlist is empty")
        allowed = create_egress_policy(allow=[{"provider": "*", "model": "*", "region": "*", "endpoint": "*"}])
        record = await allowed.assert_call({"provider": "openai", "model": "gpt-6-luna", "region": "us", "endpoint": "https://api.openai.com/v1"})
        self.assertEqual(record.decision, "allowed")

    async def test_conflicting_region_is_denied_before_a_provider_call(self) -> None:
        model = create_mock_language_model(provider="openai", model_id="gpt-6-luna")
        model.region = "us"
        target, conflict = resolve_egress_target(model, {"region": "eu"})
        self.assertEqual(conflict, "declared region does not match the model region")
        self.assertEqual(target.region, "eu")
        policy = create_egress_policy(allow=[{"provider": "*", "model": "*", "region": "*", "endpoint": "*"}])
        with self.assertRaises(EgressDeniedError):
            await enforce_egress_policy(policy, model, {"region": "eu"})
        self.assertEqual(model.calls, 0)

    def test_latam_detectors_keep_invalid_check_digits(self) -> None:
        policy = create_latam_sensitive_data_policy()
        redacted = policy.redact_text("Customer 20-12345678-6 and 20-12345678-0 paid with 4242424242424242.")
        self.assertIn("[REDACTED:CUIT]", redacted)
        self.assertIn("20-12345678-0", redacted)
        self.assertIn("[REDACTED:CARD]", redacted)
        self.assertNotIn("4242424242424242", redacted)
        with self.assertRaises(TypeError):
            create_latam_sensitive_data_policy(countries=["ZZ"])


class DecisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_decision_model_validates_locally_and_reads_usage(self) -> None:
        captured: dict[str, object] = {}

        class Response:
            status_code = 200

            def json(self):
                return {
                    "model": "gpt-6-luna",
                    "answers": [{"name": "urgent", "type": "predicate", "probability": 0.25}],
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 1,
                        "total_tokens": 4,
                        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                }

        async def fetch(url, **kwargs):
            captured["url"] = url
            captured["body"] = kwargs["body"]
            return Response()

        with self.assertRaises(ConfigurationError):
            create_openai_decision_model("gpt-6-sol", api_key="test-key", base_url="https://api.openai.com/v1")
        model = create_openai_decision_model("gpt-6-luna", api_key="test-key", base_url="https://api.openai.com/v1", fetch=fetch)
        result = await model.decide(DecisionInput(input="Customer reports duplicate billing.", questions={"urgent": {"type": "predicate", "instructions": "Does this need immediate attention?"}}))
        self.assertEqual(result.answers["urgent"]["probability"], 0.25)
        self.assertEqual(result.usage["cache_write_tokens"], 0)
        self.assertEqual(captured["url"], "https://api.openai.com/v1/decisions")

    async def test_gateway_decide_falls_back_only_on_the_explicit_status_list(self) -> None:
        capabilities = DecisionCapabilities(("predicate",), ("text",), ("string",), True, True)

        class Destination:
            def __init__(self, name: str, *, fail: bool) -> None:
                self.provider = "openai"
                self.model_id = "gpt-6-luna"
                self.endpoint = f"https://example.test/{name}"
                self.capabilities = capabilities
                self.fail = fail
                self.calls = 0

            def validate(self, decision_input: DecisionInput) -> None:
                from zhivex_ai.decisions import validate_decision

                validate_decision(decision_input, self.capabilities)

            async def decide(self, decision_input: DecisionInput) -> DecisionResult:
                self.calls += 1
                if self.fail:
                    raise ProviderHTTPError("busy", 429)
                return DecisionResult(
                    answers={"urgent": {"type": "predicate", "probability": 0.5}},
                    provenance={"provider": self.provider, "model_id": self.model_id, "endpoint": self.endpoint, "protocol": "openai.decisions"},
                    usage={"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
                )

        primary = Destination("primary", fail=True)
        backup = Destination("backup", fail=False)
        decide = create_gateway_decide({
            "primary": GatewayDecisionTarget(primary, reserve_usd=0.1),
            "backup": GatewayDecisionTarget(backup, reserve_usd=0.2),
        })
        result = await decide(
            primary="primary",
            alternatives=["backup"],
            max_attempts=2,
            fallback_on=[429],
            max_reserved_usd=0.3,
            input="Customer reports duplicate billing.",
            questions={"urgent": {"type": "predicate", "instructions": "Does this need immediate attention?"}},
        )
        self.assertEqual(result.routing["target"], "backup")
        self.assertEqual(result.routing["attempts"], 2)
        self.assertEqual(primary.calls, 1)
        self.assertEqual(backup.calls, 1)
        with self.assertRaises(ConfigurationError):
            await decide(primary="missing", input="hello", questions={"urgent": {"type": "predicate", "instructions": "Now?"}})


if __name__ == "__main__":
    unittest.main()
