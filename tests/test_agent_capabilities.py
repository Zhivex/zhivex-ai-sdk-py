from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import zhivex_ai
from zhivex_ai import (
    AgentCapabilities,
    ModelCapabilities,
    create_anthropic,
    create_azure_openai,
    create_bedrock,
    create_deepseek,
    create_gemini,
    create_kimi,
    create_ollama,
    create_openai,
    create_openrouter,
    create_qwen,
    create_vllm,
    create_vertex,
    get_agent_capabilities,
    get_agent_support_tier,
)


class _FakeBedrockClient:
    async def converse(self, payload: dict[str, object]) -> dict[str, object]:
        raise NotImplementedError


class _FakeModel:
    def __init__(self, capabilities: ModelCapabilities) -> None:
        self.capabilities = capabilities


class AgentCapabilitiesTests(TestCase):
    def test_get_agent_capabilities_returns_explicit_values(self) -> None:
        model = _FakeModel(
            ModelCapabilities(
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
                reasoning=False,
                web_search=False,
                agent_capabilities=AgentCapabilities(
                    support_tier="tier-b",
                    tool_choice_none=True,
                    hosted_web_search=True,
                    toolsets=True,
                ),
            )
        )

        capabilities = get_agent_capabilities(model)
        self.assertEqual(capabilities.support_tier, "tier-b")
        self.assertTrue(capabilities.tool_choice_none)
        self.assertTrue(capabilities.hosted_web_search)
        self.assertTrue(capabilities.toolsets)

        capabilities.support_tier = "tier-a"
        self.assertEqual(get_agent_support_tier(model), "tier-b")

    def test_get_agent_capabilities_defaults_when_missing(self) -> None:
        model = _FakeModel(
            ModelCapabilities(
                streaming=True,
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
                reasoning=False,
                web_search=False,
            )
        )

        capabilities = get_agent_capabilities(model)
        self.assertEqual(capabilities, AgentCapabilities())
        self.assertEqual(get_agent_support_tier(model), "tier-c")

    def test_top_level_beta_symbols_are_exported(self) -> None:
        for symbol in ("AgentCapabilities", "AgentSupportTier", "get_agent_capabilities", "get_agent_support_tier"):
            self.assertIn(symbol, zhivex_ai.__all__)
            self.assertTrue(hasattr(zhivex_ai, symbol))

    def test_provider_agent_support_tiers_match_initial_mapping(self) -> None:
        providers = [
            (create_openai(api_key="test"), "tier-a"),
            (create_azure_openai(api_key="test", endpoint="https://example.openai.azure.com"), "tier-a"),
            (create_anthropic(api_key="test"), "tier-b"),
            (create_gemini(api_key="test"), "tier-b"),
            (create_vertex(access_token="test", project_id="project"), "tier-b"),
            (create_bedrock(client=_FakeBedrockClient()), "tier-b"),
            (create_deepseek(api_key="test"), "tier-b"),
            (create_openrouter(api_key="test"), "tier-c"),
            (create_qwen(api_key="test"), "tier-b"),
            (create_kimi(api_key="test"), "tier-b"),
            (create_ollama(), "tier-c"),
            (create_vllm(api_key="test"), "tier-b"),
        ]

        for provider, expected_tier in providers:
            self.assertEqual(get_agent_support_tier(provider.native.language_model("test-model")), expected_tier)
