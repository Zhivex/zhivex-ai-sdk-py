from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    GatewayConfig,
    GatewayError,
    GatewayImageAttachment,
    GatewayMessage,
    GatewayModelTarget,
    ModelCatalog,
    ModelCatalogEntry,
    create_gateway,
    create_model_catalog,
)
from zhivex_ai.catalog import ModelPricing
from zhivex_ai.errors import ProviderHTTPError
from zhivex_ai.gateway import supports_vision_input
from zhivex_ai.messages import create_text_message
from zhivex_ai.providers.base import ProviderAdapter
from zhivex_ai.types import (
    GenerateResult,
    ModelCapabilities,
    ModelGenerateInput,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    TokenUsage,
)


class FailingModel:
    provider = "openai"
    model_id = "bad"
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

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        raise RuntimeError("503 upstream failed")

    async def stream(self, input: ModelGenerateInput):
        raise RuntimeError("not used")


class WorkingModel:
    provider = "anthropic"
    model_id = "good"
    capabilities = FailingModel.capabilities

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        return GenerateResult(
            messages=[create_text_message("assistant", "fallback ok")],
            text="fallback ok",
        )

    async def stream(self, input: ModelGenerateInput):
        raise RuntimeError("not used")


class RecordingModel(WorkingModel):
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        self.generate_calls += 1
        return await super().generate(input)


class RefusalModel(WorkingModel):
    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        return GenerateResult(
            messages=[create_text_message("assistant", "I cannot help with that.")],
            text="I cannot help with that.",
            finish_reason="refusal",
            provider_finish_reason="refusal",
        )


class NoVisionModel(WorkingModel):
    capabilities = ModelCapabilities(
        streaming=True,
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
        reasoning=True,
        web_search=False,
    )


class TimeoutModel(WorkingModel):
    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        raise asyncio.TimeoutError()


class RateLimitedModel(WorkingModel):
    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        raise ProviderHTTPError("OpenAI request failed with status 429.", 429)


class NonRetryableHTTPModel(WorkingModel):
    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        raise ProviderHTTPError(
            "OpenAI request failed with status 503.", 503, retryable=False
        )


class SecretHTTPModel(WorkingModel):
    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        raise ProviderHTTPError(
            "OpenAI request failed with status 400.",
            400,
            response_body='{"api_key":"sk-live-secret","message":"Bad schema"}',
        )


class SensitiveErrorModel(WorkingModel):
    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        raise RuntimeError("prompt=do not log this prompt token=super-secret-value")


class StreamingModel(WorkingModel):
    async def stream(self, input: ModelGenerateInput):
        async def generator():
            yield StreamTextDeltaEvent(text_delta='{"status":"ok"}')
            yield StreamFinishEvent(
                finish_reason="stop",
                usage=TokenUsage(input_tokens=2, output_tokens=2, total_tokens=4),
            )

        return generator()


class ObjectModel(WorkingModel):
    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        return GenerateResult(
            messages=[create_text_message("assistant", '{"status":"ok"}')],
            text='{"status":"ok"}',
        )


class Payload(BaseModel):
    status: str


class GatewayTests(IsolatedAsyncioTestCase):
    def test_deepseek_gateway_targets_are_text_only(self) -> None:
        self.assertFalse(supports_vision_input("deepseek", "deepseek-v4-flash"))
        self.assertFalse(supports_vision_input("deepseek", "deepseek-v4-pro"))

    async def test_gateway_falls_back_to_second_provider(self) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: FailingModel(),
                    ),
                    "anthropic": ProviderAdapter(
                        name="anthropic",
                        language_model_factory=lambda model_id: WorkingModel(),
                    ),
                },
                max_retries=0,
                latency_bias_ms={"anthropic": 1000},
            )
        )
        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            fallbacks=[
                GatewayModelTarget(provider="anthropic", model_id="claude-3-5-sonnet")
            ],
        )
        self.assertEqual(result.text, "fallback ok")
        self.assertEqual(result.provider_used, "anthropic")
        self.assertEqual(len(result.attempts), 2)

    async def test_gateway_keeps_primary_first_even_if_fallback_scores_higher(
        self,
    ) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: WorkingModel(),
                    ),
                    "anthropic": ProviderAdapter(
                        name="anthropic",
                        language_model_factory=lambda model_id: WorkingModel(),
                    ),
                },
                latency_bias_ms={"openai": 5000},
            )
        )
        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            fallbacks=[
                GatewayModelTarget(provider="anthropic", model_id="claude-3-5-sonnet")
            ],
            routing_mode="quality",
            task_intent="reasoning",
        )
        self.assertEqual(result.provider_used, "openai")
        self.assertEqual(result.attempts[0].provider, "openai")
        self.assertEqual(result.route_decision.ordered_targets[0].provider, "openai")

    async def test_gateway_falls_back_when_primary_returns_refusal_if_enabled(
        self,
    ) -> None:
        attempt_events: list[dict[str, object]] = []
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "anthropic": ProviderAdapter(
                        name="anthropic",
                        language_model_factory=lambda model_id: RefusalModel(),
                    ),
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: WorkingModel(),
                    ),
                },
                max_retries=0,
                fallback_on_refusal=True,
                on_attempt=lambda payload: attempt_events.append(payload),
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="anthropic", model_id="claude-fable-5"),
            fallbacks=[GatewayModelTarget(provider="openai", model_id="gpt-5.4-mini")],
        )

        self.assertEqual(result.text, "fallback ok")
        self.assertEqual(result.provider_used, "openai")
        self.assertFalse(result.attempts[0].ok)
        self.assertIn("refusal", result.attempts[0].error_message or "")
        self.assertEqual([event["ok"] for event in attempt_events], [False, True])
        self.assertEqual(attempt_events[0]["errorType"], "refusal")
        self.assertEqual(attempt_events[0]["reason"], "provider_refusal")

    async def test_gateway_returns_refusal_by_default_without_trying_fallbacks(
        self,
    ) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "anthropic": ProviderAdapter(
                        name="anthropic",
                        language_model_factory=lambda model_id: RefusalModel(),
                    ),
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: WorkingModel(),
                    ),
                },
                max_retries=0,
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="anthropic", model_id="claude-fable-5"),
            fallbacks=[GatewayModelTarget(provider="openai", model_id="gpt-5.4-mini")],
        )

        self.assertEqual(result.provider_used, "anthropic")
        self.assertEqual(result.finish_reason, "refusal")
        self.assertEqual(result.provider_finish_reason, "refusal")
        self.assertFalse(result.attempts[0].ok)
        self.assertEqual(result.attempts[0].error_type, "refusal")
        self.assertEqual(len(result.attempts), 1)

    async def test_gateway_can_disable_fallback_on_refusal(self) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "anthropic": ProviderAdapter(
                        name="anthropic",
                        language_model_factory=lambda model_id: RefusalModel(),
                    ),
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: WorkingModel(),
                    ),
                },
                max_retries=0,
                fallback_on_refusal=False,
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="anthropic", model_id="claude-fable-5"),
            fallbacks=[GatewayModelTarget(provider="openai", model_id="gpt-5.4-mini")],
        )

        self.assertEqual(result.provider_used, "anthropic")
        self.assertEqual(result.finish_reason, "refusal")
        self.assertFalse(result.attempts[0].ok)
        self.assertEqual(result.attempts[0].error_type, "refusal")

    async def test_gateway_on_attempt_emits_structured_attempt_payloads(self) -> None:
        attempt_events: list[dict[str, object]] = []

        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: FailingModel(),
                    ),
                    "anthropic": ProviderAdapter(
                        name="anthropic",
                        language_model_factory=lambda model_id: WorkingModel(),
                    ),
                },
                max_retries=0,
                on_attempt=lambda payload: attempt_events.append(payload),
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            fallbacks=[
                GatewayModelTarget(provider="anthropic", model_id="claude-3-5-sonnet")
            ],
        )

        self.assertEqual(result.provider_used, "anthropic")
        self.assertEqual(len(attempt_events), 2)
        self.assertEqual(
            [event["provider"] for event in attempt_events], ["openai", "anthropic"]
        )
        self.assertEqual([event["ok"] for event in attempt_events], [False, True])
        self.assertEqual(
            [event["attemptId"] for event in attempt_events], ["0:0", "1:0"]
        )
        self.assertEqual([event["targetRank"] for event in attempt_events], [0, 1])
        self.assertEqual([event["retry"] for event in attempt_events], [0, 0])
        self.assertTrue(all(event["phase"] == "finished" for event in attempt_events))
        self.assertTrue(all(event["terminal"] is True for event in attempt_events))
        self.assertTrue(all(int(event["latencyMs"]) >= 1 for event in attempt_events))
        self.assertEqual(attempt_events[0]["errorType"], "provider_error")
        self.assertIsNone(attempt_events[1]["errorType"])

    async def test_gateway_on_attempt_reports_non_zero_terminal_success_latency_with_controlled_clock(
        self,
    ) -> None:
        attempt_events: list[dict[str, object]] = []
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: WorkingModel(),
                    ),
                },
                max_retries=0,
                on_attempt=lambda payload: attempt_events.append(payload),
            )
        )

        with patch(
            "zhivex_ai.gateway.time.monotonic_ns",
            side_effect=[1_000_000_000, 1_000_000_001],
        ):
            result = await gateway.generate(
                messages=[GatewayMessage(role="user", content="hello")],
                primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            )

        self.assertEqual(len(attempt_events), 1)
        self.assertEqual(attempt_events[0]["latencyMs"], 1)
        self.assertEqual(result.attempts[0].latency_ms, 1)

    async def test_gateway_on_attempt_supports_async_observers(self) -> None:
        attempt_events: list[dict[str, object]] = []

        async def observe(payload: dict[str, object]) -> None:
            attempt_events.append(payload)

        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: WorkingModel(),
                    ),
                },
                max_retries=0,
                on_attempt=observe,
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
        )

        self.assertEqual(result.text, "fallback ok")
        self.assertEqual(len(attempt_events), 1)
        self.assertEqual(attempt_events[0]["ok"], True)

    async def test_gateway_observer_failures_do_not_change_provider_outcomes(
        self,
    ) -> None:
        def failing_sync_observer(payload: dict[str, object]) -> None:
            raise RuntimeError("sync telemetry unavailable")

        async def failing_async_observer(payload: dict[str, object]) -> None:
            raise RuntimeError("async telemetry unavailable")

        for observer in (failing_sync_observer, failing_async_observer):
            with self.subTest(observer=observer.__name__):
                gateway = create_gateway(
                    GatewayConfig(
                        adapters={
                            "openai": ProviderAdapter(
                                name="openai",
                                language_model_factory=lambda model_id: WorkingModel(),
                            ),
                        },
                        max_retries=0,
                        on_attempt=observer,
                    )
                )

                result = await gateway.generate(
                    messages=[GatewayMessage(role="user", content="hello")],
                    primary=GatewayModelTarget(
                        provider="openai", model_id="gpt-4o-mini"
                    ),
                )

                self.assertEqual(result.text, "fallback ok")
                self.assertEqual(len(result.attempts), 1)
                self.assertTrue(result.attempts[0].ok)

    async def test_gateway_redacts_provider_http_error_body_in_attempts(self) -> None:
        attempt_events: list[dict[str, object]] = []

        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: SecretHTTPModel(),
                    ),
                    "anthropic": ProviderAdapter(
                        name="anthropic",
                        language_model_factory=lambda model_id: WorkingModel(),
                    ),
                },
                max_retries=0,
                on_attempt=lambda payload: attempt_events.append(payload),
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            fallbacks=[
                GatewayModelTarget(provider="anthropic", model_id="claude-3-5-sonnet")
            ],
        )

        failed_message = result.attempts[0].error_message or ""
        callback_message = str(attempt_events[0]["errorMessage"])
        self.assertEqual(result.provider_used, "anthropic")
        self.assertNotIn("sk-live-secret", failed_message)
        self.assertNotIn("sk-live-secret", callback_message)
        self.assertIn("[redacted]", failed_message)
        self.assertIn("Bad schema", failed_message)
        self.assertIn("400", failed_message)
        self.assertEqual(attempt_events[0]["errorType"], "provider_http_error")

    async def test_gateway_sanitizes_generic_provider_errors_in_terminal_events(
        self,
    ) -> None:
        attempt_events: list[dict[str, object]] = []
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: SensitiveErrorModel(),
                    ),
                    "anthropic": ProviderAdapter(
                        name="anthropic",
                        language_model_factory=lambda model_id: WorkingModel(),
                    ),
                },
                max_retries=0,
                on_attempt=lambda payload: attempt_events.append(payload),
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="do not log this prompt")],
            primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            fallbacks=[
                GatewayModelTarget(provider="anthropic", model_id="claude-3-5-sonnet")
            ],
        )

        self.assertEqual(result.provider_used, "anthropic")
        self.assertEqual(attempt_events[0]["errorType"], "provider_error")
        self.assertNotIn("do not log this prompt", str(attempt_events[0]))
        self.assertNotIn("super-secret-value", str(attempt_events[0]))

    async def test_gateway_on_attempt_emits_skipped_targets(self) -> None:
        attempt_events: list[dict[str, object]] = []

        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "anthropic": ProviderAdapter(
                        name="anthropic",
                        language_model_factory=lambda model_id: WorkingModel(),
                    ),
                },
                max_retries=0,
                on_attempt=lambda payload: attempt_events.append(payload),
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            fallbacks=[
                GatewayModelTarget(provider="anthropic", model_id="claude-3-5-sonnet")
            ],
        )

        self.assertEqual(result.provider_used, "anthropic")
        self.assertEqual(attempt_events[0]["provider"], "openai")
        self.assertEqual(attempt_events[0]["ok"], False)
        self.assertEqual(attempt_events[0]["targetRank"], 0)
        self.assertIn("No adapter registered", str(attempt_events[0]["errorMessage"]))

    async def test_gateway_fail_closes_unknown_cost_under_budget_and_falls_back(
        self,
    ) -> None:
        attempt_events: list[dict[str, object]] = []
        primary_model = RecordingModel()
        fallback_model = RecordingModel()
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: primary_model,
                    ),
                    "anthropic": ProviderAdapter(
                        name="anthropic",
                        language_model_factory=lambda model_id: fallback_model,
                    ),
                },
                max_retries=0,
                on_attempt=lambda payload: attempt_events.append(payload),
                model_costs_per_1k_tokens={"anthropic": {"priced-fallback": 0.5}},
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="do not log this prompt")],
            primary=GatewayModelTarget(provider="openai", model_id="unknown-price"),
            fallbacks=[
                GatewayModelTarget(provider="anthropic", model_id="priced-fallback")
            ],
            max_cost_per_1k_tokens=1,
        )

        self.assertEqual(result.provider_used, "anthropic")
        self.assertEqual(primary_model.generate_calls, 0)
        self.assertEqual(fallback_model.generate_calls, 1)
        self.assertEqual(result.attempts[0].reason, "cost_unknown")
        self.assertEqual(attempt_events[0]["provider"], "openai")
        self.assertEqual(attempt_events[0]["modelId"], "unknown-price")
        self.assertEqual(attempt_events[0]["reason"], "cost_unknown")
        self.assertNotIn("do not log this prompt", str(attempt_events[0]))

    async def test_gateway_allows_unknown_cost_without_budget(self) -> None:
        model = RecordingModel()
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai", language_model_factory=lambda model_id: model
                    )
                },
                max_retries=0,
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="unknown-price"),
        )

        self.assertEqual(result.provider_used, "openai")
        self.assertEqual(model.generate_calls, 1)

    async def test_gateway_treats_invalid_model_cost_as_unknown_before_resolving_adapter(
        self,
    ) -> None:
        factory_calls = 0

        def create_model(model_id: str) -> RecordingModel:
            nonlocal factory_calls
            factory_calls += 1
            return RecordingModel()

        for model_id, model_cost in (
            ("negative-price", -1.0),
            ("non-finite-price", float("nan")),
        ):
            with self.subTest(model_id=model_id):
                attempt_events: list[dict[str, object]] = []
                gateway = create_gateway(
                    GatewayConfig(
                        adapters={
                            "openai": ProviderAdapter(
                                name="openai", language_model_factory=create_model
                            )
                        },
                        max_retries=0,
                        on_attempt=lambda payload: attempt_events.append(payload),
                        model_costs_per_1k_tokens={"openai": {model_id: model_cost}},
                    )
                )

                with self.assertRaises(GatewayError):
                    await gateway.generate(
                        messages=[GatewayMessage(role="user", content="hello")],
                        primary=GatewayModelTarget(
                            provider="openai", model_id=model_id
                        ),
                        required_capabilities={"structured_output": True},
                        max_cost_per_1k_tokens=1,
                    )

                self.assertEqual(factory_calls, 0)
                self.assertEqual(attempt_events[0]["reason"], "cost_unknown")

    async def test_gateway_accepts_cost_less_than_or_equal_to_budget(self) -> None:
        for model_id, model_cost in (("under-budget", 0.5), ("at-budget", 1.0)):
            with self.subTest(model_id=model_id):
                model = RecordingModel()
                gateway = create_gateway(
                    GatewayConfig(
                        adapters={
                            "openai": ProviderAdapter(
                                name="openai", language_model_factory=lambda _: model
                            )
                        },
                        max_retries=0,
                        model_costs_per_1k_tokens={"openai": {model_id: model_cost}},
                    )
                )

                result = await gateway.generate(
                    messages=[GatewayMessage(role="user", content="hello")],
                    primary=GatewayModelTarget(provider="openai", model_id=model_id),
                    max_cost_per_1k_tokens=1,
                )

                self.assertEqual(result.provider_used, "openai")
                self.assertEqual(model.generate_calls, 1)

    async def test_gateway_skips_cost_above_budget_with_typed_reason(self) -> None:
        primary_model = RecordingModel()
        fallback_model = RecordingModel()
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: primary_model,
                    ),
                    "anthropic": ProviderAdapter(
                        name="anthropic",
                        language_model_factory=lambda model_id: fallback_model,
                    ),
                },
                provider_costs_per_1k_tokens={"openai": 0.1},
                max_retries=0,
                model_costs_per_1k_tokens={
                    "openai": {"over-budget": 1.01},
                    "anthropic": {"priced-fallback": 0.5},
                },
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="over-budget"),
            fallbacks=[
                GatewayModelTarget(provider="anthropic", model_id="priced-fallback")
            ],
            max_cost_per_1k_tokens=1,
        )

        self.assertEqual(result.provider_used, "anthropic")
        self.assertEqual(primary_model.generate_calls, 0)
        self.assertEqual(fallback_model.generate_calls, 1)
        self.assertEqual(result.attempts[0].reason, "cost_exceeds_budget")

    async def test_gateway_uses_model_costs_to_rank_fallbacks(self) -> None:
        cheap_model = RecordingModel()
        expensive_model = RecordingModel()
        models = {
            "primary-fails": FailingModel(),
            "expensive-fallback": expensive_model,
            "cheap-fallback": cheap_model,
        }
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: models[model_id],
                    )
                },
                max_retries=0,
                model_costs_per_1k_tokens={
                    "openai": {
                        "primary-fails": 0.5,
                        "expensive-fallback": 2,
                        "cheap-fallback": 0.5,
                    }
                },
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="primary-fails"),
            fallbacks=[
                GatewayModelTarget(provider="openai", model_id="expensive-fallback"),
                GatewayModelTarget(provider="openai", model_id="cheap-fallback"),
            ],
            routing_mode="quality",
        )

        self.assertEqual(result.model_used, "cheap-fallback")
        self.assertEqual(cheap_model.generate_calls, 1)
        self.assertEqual(expensive_model.generate_calls, 0)

    async def test_gateway_catalog_changes_routing_without_model_name_heuristics(
        self,
    ) -> None:
        async def run_with(catalog: ModelCatalog | None):
            deceptive = RecordingModel()
            explicit_speed = RecordingModel()
            models = {
                "primary-fails": FailingModel(),
                "ultra-pro-flash-lite": deceptive,
                "plain-model": explicit_speed,
            }
            gateway = create_gateway(
                GatewayConfig(
                    adapters={
                        "openai": ProviderAdapter(
                            name="openai",
                            language_model_factory=lambda model_id: models[model_id],
                        )
                    },
                    model_catalog=catalog,
                    max_retries=0,
                )
            )
            return await gateway.generate(
                messages=[GatewayMessage(role="user", content="hello")],
                primary=GatewayModelTarget(provider="openai", model_id="primary-fails"),
                fallbacks=[
                    GatewayModelTarget(
                        provider="openai", model_id="ultra-pro-flash-lite"
                    ),
                    GatewayModelTarget(provider="openai", model_id="plain-model"),
                ],
                routing_mode="speed",
            )

        legacy = await run_with(None)
        self.assertEqual(legacy.model_used, "ultra-pro-flash-lite")

        catalog = create_model_catalog(
            [
                ModelCatalogEntry(
                    provider="openai",
                    model_id="primary-fails",
                    recommended_for=["chat"],
                ),
                ModelCatalogEntry(
                    provider="openai",
                    model_id="ultra-pro-flash-lite",
                    recommended_for=["chat"],
                ),
                ModelCatalogEntry(
                    provider="openai",
                    model_id="canonical-speed-model",
                    aliases=["plain-model"],
                    cost_per_1k_tokens=0.25,
                    recommended_for=["speed"],
                    availability="preview",
                ),
            ]
        )
        catalog_driven = await run_with(catalog)

        self.assertEqual(catalog_driven.model_used, "plain-model")
        self.assertEqual(
            [
                target.model_id
                for target in catalog_driven.route_decision.ordered_targets
            ],
            ["primary-fails", "plain-model", "ultra-pro-flash-lite"],
        )
        evidence = catalog_driven.route_decision.target_evidence[1]
        self.assertEqual(evidence.canonical_model_id, "canonical-speed-model")
        self.assertEqual(evidence.scoring_source, "model_catalog")
        self.assertEqual(evidence.recommended_for, ("speed",))
        self.assertEqual(evidence.capabilities, {})
        self.assertEqual(evidence.availability, "preview")
        self.assertEqual(evidence.cost_per_1k_tokens, 0.25)
        self.assertEqual(evidence.cost_source, "model_catalog")
        self.assertIn(
            "Catalog metadata used for 3 target(s)",
            catalog_driven.route_decision.reason,
        )

    async def test_gateway_catalog_capabilities_fail_closed_when_required_metadata_is_missing(
        self,
    ) -> None:
        missing_capability_model = RecordingModel()
        explicit_capability_model = RecordingModel()
        models = {
            "looks-capable-pro": missing_capability_model,
            "explicit-capabilities": explicit_capability_model,
        }
        catalog = create_model_catalog(
            [
                ModelCatalogEntry(
                    provider="openai",
                    model_id="looks-capable-pro",
                    recommended_for=["chat", "tools"],
                ),
                ModelCatalogEntry(
                    provider="openai",
                    model_id="explicit-capabilities",
                    recommended_for=["chat"],
                    capabilities=FailingModel.capabilities,
                ),
            ]
        )
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: models[model_id],
                    )
                },
                model_catalog=catalog,
                max_retries=0,
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="looks-capable-pro"),
            fallbacks=[
                GatewayModelTarget(provider="openai", model_id="explicit-capabilities")
            ],
            required_capabilities={"structuredOutput": True},
            routing_mode="quality",
        )

        self.assertEqual(result.model_used, "explicit-capabilities")
        self.assertEqual(missing_capability_model.generate_calls, 0)
        self.assertEqual(explicit_capability_model.generate_calls, 1)
        self.assertEqual(result.attempts[0].reason, "capability_mismatch")
        self.assertEqual(
            result.route_decision.required_capabilities, ("structuredOutput",)
        )
        self.assertNotIn(
            "structured_output", result.route_decision.target_evidence[0].capabilities
        )
        self.assertTrue(
            result.route_decision.target_evidence[1].capabilities["structured_output"]
        )

    async def test_gateway_uses_source_backed_catalog_pricing_and_exposes_provenance(
        self,
    ) -> None:
        model = RecordingModel()
        catalog = create_model_catalog(
            [
                ModelCatalogEntry(
                    provider="openai",
                    model_id="priced-model",
                    recommended_for=("chat",),
                    pricing=ModelPricing(
                        currency="USD",
                        source_url="https://example.test/pricing",
                        input_per_1m_tokens=5,
                        output_per_1m_tokens=30,
                        effective_from="2026-08-01",
                        effective_until="2026-08-31",
                    ),
                )
            ]
        )
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai", language_model_factory=lambda _: model
                    )
                },
                model_catalog=catalog,
                max_retries=0,
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="priced-model"),
            max_cost_per_1k_tokens=0.03,
        )

        evidence = result.route_decision.target_evidence[0]
        self.assertEqual(evidence.cost_per_1k_tokens, 0.03)
        self.assertEqual(evidence.cost_source, "model_catalog")
        self.assertEqual(evidence.pricing_currency, "USD")
        self.assertEqual(evidence.pricing_source_url, "https://example.test/pricing")
        self.assertEqual(evidence.pricing_effective_until, "2026-08-31")

    async def test_gateway_treats_expired_catalog_pricing_as_unknown(self) -> None:
        model = RecordingModel()
        catalog = create_model_catalog(
            [
                ModelCatalogEntry(
                    provider="openai",
                    model_id="expired-price",
                    pricing=ModelPricing(
                        currency="USD",
                        source_url="https://example.test/pricing",
                        input_per_1m_tokens=1,
                        output_per_1m_tokens=2,
                        effective_until="2000-01-01",
                    ),
                )
            ]
        )
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai", language_model_factory=lambda _: model
                    )
                },
                model_catalog=catalog,
                max_retries=0,
            )
        )

        with self.assertRaises(GatewayError):
            await gateway.generate(
                messages=[GatewayMessage(role="user", content="hello")],
                primary=GatewayModelTarget(provider="openai", model_id="expired-price"),
                max_cost_per_1k_tokens=1,
            )

        self.assertEqual(model.generate_calls, 0)

    async def test_gateway_skips_retired_and_non_language_catalog_targets(self) -> None:
        retired = RecordingModel()
        image = RecordingModel()
        current = RecordingModel()
        models = {"retired": retired, "image": image, "current": current}
        catalog = create_model_catalog(
            [
                ModelCatalogEntry(
                    provider="openai",
                    model_id="retired",
                    availability="retired",
                    replacement_model_id="current",
                ),
                ModelCatalogEntry(
                    provider="openai", model_id="image", api_surface="image"
                ),
                ModelCatalogEntry(provider="openai", model_id="current"),
            ]
        )
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: models[model_id],
                    )
                },
                model_catalog=catalog,
                max_retries=0,
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="retired"),
            fallbacks=[
                GatewayModelTarget(provider="openai", model_id="image"),
                GatewayModelTarget(provider="openai", model_id="current"),
            ],
        )

        self.assertEqual(result.model_used, "current")
        self.assertEqual(
            [attempt.reason for attempt in result.attempts[:2]],
            ["model_unavailable", "unsupported_api_surface"],
        )
        self.assertEqual(retired.generate_calls, 0)
        self.assertEqual(image.generate_calls, 0)
        self.assertEqual(current.generate_calls, 1)

    async def test_gateway_model_cost_precedes_provider_default(self) -> None:
        model = RecordingModel()
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai", language_model_factory=lambda model_id: model
                    )
                },
                provider_costs_per_1k_tokens={"openai": 10},
                max_retries=0,
                model_costs_per_1k_tokens={"openai": {"cheap-model": 0.5}},
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="cheap-model"),
            max_cost_per_1k_tokens=1,
        )

        self.assertEqual(result.provider_used, "openai")
        self.assertEqual(model.generate_calls, 1)

    async def test_gateway_catalog_model_cost_precedes_provider_default_and_resolves_alias(
        self,
    ) -> None:
        model = RecordingModel()
        catalog = create_model_catalog(
            [
                ModelCatalogEntry(
                    provider="openai",
                    model_id="canonical-model",
                    aliases=["model-alias"],
                    cost_per_1k_tokens=0.5,
                )
            ]
        )
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai", language_model_factory=lambda model_id: model
                    )
                },
                model_catalog=catalog,
                provider_costs_per_1k_tokens={"openai": 10},
                max_retries=0,
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="model-alias"),
            max_cost_per_1k_tokens=1,
        )

        self.assertEqual(result.provider_used, "openai")
        self.assertEqual(model.generate_calls, 1)

    async def test_gateway_provider_cost_fallback_remains_compatible(self) -> None:
        model = RecordingModel()
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai", language_model_factory=lambda model_id: model
                    )
                },
                provider_costs_per_1k_tokens={"openai": 0.5},
                max_retries=0,
            )
        )

        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(
                provider="openai", model_id="legacy-provider-price"
            ),
            max_cost_per_1k_tokens=1,
        )

        self.assertEqual(result.provider_used, "openai")
        self.assertEqual(model.generate_calls, 1)

    async def test_gateway_can_fail_fast_when_adapter_is_missing(self) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "anthropic": ProviderAdapter(
                        name="anthropic",
                        language_model_factory=lambda model_id: WorkingModel(),
                    ),
                },
                fail_on_missing_adapter=True,
            )
        )

        with self.assertRaises(GatewayError) as context:
            await gateway.generate(
                messages=[GatewayMessage(role="user", content="hello")],
                primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
                fallbacks=[
                    GatewayModelTarget(
                        provider="anthropic", model_id="claude-3-5-sonnet"
                    )
                ],
            )

        self.assertFalse(context.exception.retryable)
        self.assertIn(
            'No adapter registered for provider "openai"', str(context.exception)
        )

    async def test_gateway_skips_non_vision_targets_instead_of_dropping_images(
        self,
    ) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: NoVisionModel(),
                    ),
                    "anthropic": ProviderAdapter(
                        name="anthropic",
                        language_model_factory=lambda model_id: WorkingModel(),
                    ),
                },
                max_retries=0,
            )
        )
        result = await gateway.generate(
            messages=[
                GatewayMessage(
                    role="user",
                    content="describe this",
                    images=[
                        GatewayImageAttachment(
                            data_url="data:image/png;base64,aaaa", mime_type="image/png"
                        )
                    ],
                )
            ],
            primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            fallbacks=[
                GatewayModelTarget(provider="anthropic", model_id="claude-3-5-sonnet")
            ],
        )
        self.assertEqual(result.provider_used, "anthropic")
        self.assertEqual(result.attempts[0].provider, "openai")
        self.assertIn(
            "does not support vision input", result.attempts[0].error_message or ""
        )

    async def test_gateway_timeout_error_has_clear_message_and_retryable_flag(
        self,
    ) -> None:
        attempt_events: list[dict[str, object]] = []
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: TimeoutModel(),
                    ),
                },
                max_retries=0,
                attempt_timeout_ms=3210,
                on_attempt=lambda payload: attempt_events.append(payload),
            )
        )
        with self.assertRaises(GatewayError) as context:
            await gateway.generate(
                messages=[GatewayMessage(role="user", content="hello")],
                primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            )
        self.assertTrue(context.exception.retryable)
        self.assertIn(
            'timed out for "openai/gpt-4o-mini" after 3210 ms', str(context.exception)
        )
        self.assertEqual(len(attempt_events), 1)
        self.assertEqual(attempt_events[0]["ok"], False)
        self.assertEqual(attempt_events[0]["errorType"], "timeout")
        self.assertEqual(attempt_events[0]["attemptId"], "0:0")

    async def test_gateway_retry_events_keep_deterministic_indexes_and_cardinality(
        self,
    ) -> None:
        attempt_events: list[dict[str, object]] = []
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: FailingModel(),
                    ),
                },
                max_retries=1,
                retry_backoff_ms=0,
                on_attempt=lambda payload: attempt_events.append(payload),
            )
        )

        with self.assertRaises(GatewayError):
            await gateway.generate(
                messages=[GatewayMessage(role="user", content="hello")],
                primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            )

        self.assertEqual(len(attempt_events), 2)
        self.assertEqual(
            [event["attemptId"] for event in attempt_events], ["0:0", "0:1"]
        )
        self.assertEqual([event["retry"] for event in attempt_events], [0, 1])
        self.assertEqual([event["targetRank"] for event in attempt_events], [0, 0])
        self.assertTrue(all(event["ok"] is False for event in attempt_events))
        self.assertTrue(all(event["phase"] == "finished" for event in attempt_events))

    async def test_gateway_provider_http_error_preserves_retryable_status(self) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: RateLimitedModel(),
                    ),
                },
                max_retries=0,
            )
        )
        with self.assertRaises(GatewayError) as context:
            await gateway.generate(
                messages=[GatewayMessage(role="user", content="hello")],
                primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            )
        self.assertTrue(context.exception.retryable)
        self.assertIn("429", str(context.exception))

    async def test_gateway_respects_explicit_non_retryable_http_error(self) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: NonRetryableHTTPModel(),
                    ),
                },
                max_retries=0,
            )
        )
        with self.assertRaises(GatewayError) as context:
            await gateway.generate(
                messages=[GatewayMessage(role="user", content="hello")],
                primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            )
        self.assertFalse(context.exception.retryable)
        self.assertIn("503", str(context.exception))

    async def test_gateway_stream_text_collects_result(self) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: StreamingModel(),
                    ),
                },
                max_retries=0,
            )
        )
        result = gateway.stream_text(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
        )
        final = await result.collect()
        self.assertEqual(final.text, '{"status":"ok"}')
        self.assertEqual(final.provider_used, "openai")

    async def test_gateway_generate_object_enriches_gateway_metadata(self) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai",
                        language_model_factory=lambda model_id: ObjectModel(),
                    ),
                },
                max_retries=0,
            )
        )
        result = await gateway.generate_object(
            schema=Payload,
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
        )
        self.assertEqual(result.object.status, "ok")
        self.assertEqual(result.provider_used, "openai")
