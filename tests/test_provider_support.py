from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    UnsupportedFeatureError,
    create_anthropic,
    create_azure_openai,
    create_bedrock,
    create_gemini,
    create_kimi,
    create_ollama,
    create_openai,
    create_openrouter,
    create_qwen,
    create_vllm,
    create_vertex,
)
from zhivex_ai.provider_support import (
    TIER_1_PROVIDERS,
    build_provider_support_rows,
    get_tier_1_provider_rows,
    render_provider_support_markdown,
)


class _FakeBedrockClient:
    async def converse(self, payload: dict[str, object]) -> dict[str, object]:
        raise NotImplementedError


class ProviderSupportTests(TestCase):
    def test_build_provider_support_rows_reports_portable_tiers(self) -> None:
        rows = build_provider_support_rows(
            [
                create_openai(api_key="test"),
                create_azure_openai(api_key="test", endpoint="https://example.openai.azure.com"),
                create_anthropic(api_key="test"),
                create_gemini(api_key="test"),
                create_vertex(access_token="test", project_id="project"),
                create_bedrock(client=_FakeBedrockClient()),
                create_openrouter(api_key="test"),
                create_qwen(api_key="test"),
                create_kimi(api_key="test"),
                create_ollama(),
                create_vllm(api_key="test"),
            ]
        )
        tiers = {row.provider: (row.tier, row.portable_badge) for row in rows}
        native = {row.provider: row.native_support for row in rows}
        agent = {row.provider: row.agent_capabilities for row in rows}

        self.assertEqual(tiers["openai"], ("portable", True))
        self.assertEqual(tiers["anthropic"], ("portable", True))
        self.assertEqual(tiers["azure-openai"], ("portable", True))
        self.assertEqual(tiers["gemini"], ("portable", True))
        self.assertEqual(tiers["vertex"], ("portable", True))
        self.assertEqual(tiers["bedrock"], ("native-only", False))
        self.assertEqual(tiers["openrouter"], ("native-only", False))
        self.assertEqual(tiers["qwen"], ("compatibility", False))
        self.assertEqual(tiers["kimi"], ("compatibility", False))
        self.assertEqual(tiers["ollama"], ("compatibility", False))
        self.assertEqual(tiers["vllm"], ("portable", True))
        self.assertFalse(rows[[row.provider for row in rows].index("kimi")].portable_support.embeddings)
        self.assertTrue(native["kimi"].files)
        self.assertTrue(native["kimi"].batches)
        self.assertTrue(native["vllm"].realtime)
        self.assertTrue(native["vllm"].transcription)
        self.assertFalse(native["kimi"].responses)
        self.assertTrue(native["openai"].images)
        self.assertTrue(native["openai"].uploads)
        self.assertTrue(native["openai"].moderations)
        self.assertTrue(native["openai"].batches)
        self.assertTrue(native["openai"].containers)
        self.assertTrue(native["openai"].skills)
        self.assertTrue(native["anthropic"].files)
        self.assertTrue(native["gemini"].images)
        self.assertTrue(native["gemini"].videos)
        self.assertTrue(native["gemini"].media)
        self.assertTrue(native["gemini"].batches)
        self.assertTrue(native["gemini"].interactions)
        self.assertTrue(native["vertex"].images)
        self.assertTrue(native["vertex"].videos)
        self.assertTrue(native["vertex"].media)
        self.assertFalse(native["vertex"].batches)
        self.assertFalse(native["vertex"].interactions)
        self.assertEqual(agent["openai"].support_tier, "tier-a")
        self.assertTrue(agent["openai"].approval_requests)
        self.assertTrue(agent["openai"].remote_mcp)
        self.assertEqual(agent["anthropic"].support_tier, "tier-b")
        self.assertTrue(agent["anthropic"].code_execution)
        self.assertTrue(agent["anthropic"].toolsets)
        self.assertTrue(agent["gemini"].hosted_file_search)
        self.assertTrue(agent["gemini"].computer_use)
        self.assertTrue(agent["vertex"].hosted_file_search)
        self.assertTrue(agent["vertex"].computer_use)
        self.assertEqual(agent["bedrock"].support_tier, "tier-c")
        self.assertFalse(agent["bedrock"].tool_choice_none)

        markdown = render_provider_support_markdown(rows)
        self.assertIn("### Tier-1 Providers", markdown)
        self.assertIn("- `openai`", markdown)
        self.assertIn("- `vertex`", markdown)
        self.assertIn("- `vllm`", markdown)
        self.assertIn("### Portable Support", markdown)
        self.assertIn("| openai | portable | Yes |", markdown)
        self.assertIn("| vllm | portable | Yes |", markdown)
        self.assertIn("| anthropic | portable | Yes |", markdown)
        self.assertIn("### Native Extras", markdown)
        self.assertIn("### Agent Capabilities", markdown)
        self.assertIn("| openai | tier-a | Yes | Yes | Yes | Yes | Yes | Yes | No | No |", markdown)
        self.assertIn("| anthropic | tier-b | Yes | No | Yes | No | No | No | Yes | Yes |", markdown)

    def test_tier_1_provider_contract_is_explicit(self) -> None:
        rows = build_provider_support_rows(
            [
                create_openai(api_key="test"),
                create_azure_openai(api_key="test", endpoint="https://example.openai.azure.com"),
                create_anthropic(api_key="test"),
                create_gemini(api_key="test"),
                create_vertex(access_token="test", project_id="project"),
                create_vllm(api_key="test"),
            ]
        )

        self.assertEqual(TIER_1_PROVIDERS, ("openai", "anthropic", "azure-openai", "gemini", "vertex", "vllm"))
        tier_1_rows = get_tier_1_provider_rows(rows)

        self.assertEqual([row.provider for row in tier_1_rows], list(TIER_1_PROVIDERS))
        self.assertTrue(all(row.tier == "portable" for row in tier_1_rows))
        self.assertTrue(all(row.portable_badge for row in tier_1_rows))

    def test_non_portable_provider_rejects_portable_model_construction(self) -> None:
        provider = create_bedrock(client=_FakeBedrockClient())

        with self.assertRaises(UnsupportedFeatureError):
            provider("anthropic.claude-sonnet-4")

        native_model = provider.native.language_model("anthropic.claude-sonnet-4")
        self.assertEqual(native_model.provider, "bedrock")
