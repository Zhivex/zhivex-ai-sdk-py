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
    create_vertex,
)
from zhivex_ai.provider_support import build_provider_support_rows, render_provider_support_markdown


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
            ]
        )
        tiers = {row.provider: (row.tier, row.portable_badge) for row in rows}

        self.assertEqual(tiers["openai"], ("portable", True))
        self.assertEqual(tiers["azure-openai"], ("portable", True))
        self.assertEqual(tiers["gemini"], ("portable", True))
        self.assertEqual(tiers["vertex"], ("portable", True))
        self.assertEqual(tiers["anthropic"], ("native-only", False))
        self.assertEqual(tiers["bedrock"], ("native-only", False))
        self.assertEqual(tiers["openrouter"], ("native-only", False))
        self.assertEqual(tiers["qwen"], ("compatibility", False))
        self.assertEqual(tiers["kimi"], ("compatibility", False))
        self.assertEqual(tiers["ollama"], ("compatibility", False))

        markdown = render_provider_support_markdown(rows)
        self.assertIn("### Portable Support", markdown)
        self.assertIn("| openai | portable | Yes |", markdown)
        self.assertIn("| anthropic | native-only | No |", markdown)
        self.assertIn("### Native Extras", markdown)

    def test_non_portable_provider_rejects_portable_model_construction(self) -> None:
        provider = create_anthropic(api_key="test")

        with self.assertRaises(UnsupportedFeatureError):
            provider("claude-sonnet-4-20250514")

        native_model = provider.native.language_model("claude-sonnet-4-20250514")
        self.assertEqual(native_model.provider, "anthropic")
