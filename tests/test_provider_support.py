from __future__ import annotations

import sys
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    UnsupportedFeatureError,
    create_anthropic,
    create_azure_openai,
    create_bedrock,
    create_deepseek,
    create_gemini,
    create_kimi,
    create_meta,
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
from zhivex_ai.types import ModelGenerateInput, StructuredOutputConfig


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
                create_deepseek(api_key="test"),
                create_openrouter(api_key="test"),
                create_qwen(api_key="test"),
                create_kimi(api_key="test"),
                create_meta(api_key="test"),
                create_ollama(),
                create_vllm(api_key="test"),
            ]
        )
        tiers = {row.provider: (row.tier, row.portable_badge) for row in rows}
        evidence = {row.provider: row.evidence_status for row in rows}
        native = {row.provider: row.native_support for row in rows}
        agent = {row.provider: row.agent_capabilities for row in rows}

        self.assertEqual(tiers["openai"], ("portable", True))
        self.assertEqual(tiers["anthropic"], ("portable", True))
        self.assertEqual(tiers["azure-openai"], ("portable", True))
        self.assertEqual(tiers["gemini"], ("portable", True))
        self.assertEqual(tiers["vertex"], ("portable", True))
        self.assertEqual(tiers["bedrock"], ("native-only", False))
        self.assertEqual(tiers["deepseek"], ("portable", True))
        self.assertEqual(tiers["openrouter"], ("native-only", False))
        self.assertEqual(tiers["qwen"], ("portable", True))
        self.assertEqual(tiers["kimi"], ("portable", True))
        self.assertEqual(tiers["meta"], ("portable", True))
        self.assertEqual(tiers["ollama"], ("compatibility", False))
        self.assertEqual(tiers["vllm"], ("portable", True))
        self.assertEqual(evidence["openai"], "contract-supported")
        self.assertEqual(evidence["meta"], "contract-supported")
        self.assertEqual(evidence["bedrock"], "experimental/native-only")
        self.assertEqual(evidence["ollama"], "experimental/native-only")
        self.assertEqual(evidence["openrouter"], "experimental/native-only")
        self.assertNotIn("release-certified", evidence.values())
        self.assertFalse(rows[[row.provider for row in rows].index("kimi")].portable_support.embeddings)
        self.assertTrue(native["kimi"].files)
        self.assertTrue(native["kimi"].batches)
        self.assertTrue(native["vllm"].realtime)
        self.assertTrue(native["vllm"].transcription)
        self.assertFalse(native["kimi"].responses)
        self.assertFalse(native["deepseek"].embeddings)
        self.assertFalse(native["deepseek"].files)
        self.assertTrue(native["openai"].images)
        self.assertTrue(native["openai"].uploads)
        self.assertTrue(native["openai"].moderations)
        self.assertTrue(native["openai"].batches)
        self.assertTrue(native["openai"].containers)
        self.assertTrue(native["openai"].skills)
        self.assertTrue(native["azure-openai"].file_search)
        self.assertTrue(native["azure-openai"].responses)
        self.assertTrue(native["azure-openai"].conversations)
        self.assertFalse(native["azure-openai"].containers)
        self.assertFalse(native["azure-openai"].skills)
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
        self.assertTrue(agent["openai"].code_execution)
        self.assertTrue(agent["anthropic"].toolsets)
        self.assertTrue(agent["gemini"].hosted_file_search)
        self.assertTrue(agent["gemini"].computer_use)
        self.assertTrue(agent["vertex"].hosted_file_search)
        self.assertTrue(agent["vertex"].computer_use)
        self.assertEqual(agent["qwen"].support_tier, "tier-b")
        self.assertTrue(agent["qwen"].remote_mcp)
        self.assertTrue(agent["qwen"].code_execution)
        self.assertTrue(native["qwen"].files)
        self.assertTrue(native["qwen"].batches)
        self.assertTrue(native["qwen"].responses)
        self.assertTrue(native["qwen"].embeddings)
        self.assertTrue(native["qwen"].transcription)
        self.assertTrue(native["qwen"].speech)
        self.assertFalse(native["qwen"].file_search)
        self.assertEqual(agent["kimi"].support_tier, "tier-b")
        self.assertTrue(agent["kimi"].toolsets)
        self.assertTrue(native["kimi"].count_tokens)
        self.assertTrue(native["kimi"].formulas)
        self.assertTrue(native["meta"].files)
        self.assertTrue(native["meta"].responses)
        self.assertFalse(native["meta"].embeddings)
        self.assertEqual(agent["deepseek"].support_tier, "tier-b")
        self.assertTrue(agent["deepseek"].tool_choice_none)
        self.assertEqual(agent["meta"].support_tier, "tier-b")
        self.assertFalse(agent["meta"].tool_choice_none)
        self.assertTrue(agent["meta"].hosted_web_search)
        self.assertTrue(agent["meta"].toolsets)
        self.assertTrue(native["bedrock"].tools)
        self.assertEqual(agent["bedrock"].support_tier, "tier-b")
        self.assertTrue(agent["bedrock"].tool_choice_none)

        markdown = render_provider_support_markdown(rows)
        self.assertIn("### Tier-1 Providers", markdown)
        self.assertIn("- `openai`", markdown)
        self.assertIn("- `vertex`", markdown)
        self.assertIn("- `vllm`", markdown)
        self.assertIn("- `deepseek`", markdown)
        self.assertIn("Tier-1 identifies shared contract coverage; it does not establish live release certification.", markdown)
        self.assertIn("### Provider Evidence", markdown)
        self.assertIn("| openai | contract-supported |", markdown)
        self.assertIn("| meta | contract-supported |", markdown)
        self.assertIn("| bedrock | experimental/native-only |", markdown)
        self.assertIn("### Portable Support", markdown)
        self.assertIn("| openai | portable | Yes |", markdown)
        self.assertIn("| vllm | portable | Yes |", markdown)
        self.assertIn("| deepseek | portable | Yes |", markdown)
        self.assertIn("| anthropic | portable | Yes |", markdown)
        self.assertIn("| meta | portable | Yes | Yes | Yes | Yes | Yes | No | No | Yes | No | No |", markdown)
        self.assertIn("| bedrock | native-only | No | N/A | N/A |", markdown)
        self.assertIn("| ollama | compatibility | No | N/A | N/A |", markdown)
        self.assertIn("| openrouter | native-only | No | N/A | N/A |", markdown)
        self.assertIn("### Native Extras", markdown)
        self.assertIn(
            "| Provider | Text | Streaming | Structured Output | Tools | Embeddings | Grounding | Transcription | Speech | Files |",
            markdown,
        )
        self.assertIn("| bedrock | Yes | Yes | No | Yes | No | No | No | No | No |", markdown)
        self.assertIn("| qwen | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |", markdown)
        self.assertIn("### Agent Capabilities", markdown)
        self.assertIn("| openai | tier-a | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No |", markdown)
        self.assertIn("| anthropic | tier-b | Yes | No | Yes | No | No | No | Yes | Yes |", markdown)
        self.assertIn("| bedrock | tier-b | Yes | No | No | No | No | No | No | No |", markdown)
        self.assertIn("| qwen | tier-b | Yes | No | Yes | Yes | Yes | No | Yes | No |", markdown)
        self.assertIn("| kimi | tier-b | Yes | No | No | No | No | No | No | Yes |", markdown)
        self.assertIn("| deepseek | tier-b | Yes | No | No | No | No | No | No | No |", markdown)
        self.assertIn("| meta | tier-b | No | No | Yes | No | No | No | No | Yes |", markdown)

    def test_tier_1_provider_contract_is_explicit(self) -> None:
        rows = build_provider_support_rows(
            [
                create_openai(api_key="test"),
                create_azure_openai(api_key="test", endpoint="https://example.openai.azure.com"),
                create_anthropic(api_key="test"),
                create_gemini(api_key="test"),
                create_vertex(access_token="test", project_id="project"),
                create_deepseek(api_key="test"),
                create_qwen(api_key="test"),
                create_kimi(api_key="test"),
                create_vllm(api_key="test"),
                create_meta(api_key="test"),
            ]
        )

        self.assertEqual(
            TIER_1_PROVIDERS,
            (
                "openai",
                "anthropic",
                "azure-openai",
                "gemini",
                "vertex",
                "qwen",
                "kimi",
                "deepseek",
                "meta",
                "vllm",
            ),
        )
        tier_1_rows = get_tier_1_provider_rows(rows)

        self.assertEqual([row.provider for row in tier_1_rows], list(TIER_1_PROVIDERS))
        self.assertIn("meta", TIER_1_PROVIDERS)
        self.assertIn("meta", [row.provider for row in tier_1_rows])
        self.assertTrue(all(row.tier == "portable" for row in tier_1_rows))
        self.assertTrue(all(row.portable_badge for row in tier_1_rows))
        self.assertTrue(all(row.evidence_status == "contract-supported" for row in tier_1_rows))

    def test_release_certification_requires_explicit_validated_evidence(self) -> None:
        providers = [
            create_openai(api_key="test"),
            create_meta(api_key="test"),
            create_bedrock(client=_FakeBedrockClient()),
        ]

        rows = build_provider_support_rows(
            providers,
            validated_release_certifications={"openai"},
        )
        evidence = {row.provider: row.evidence_status for row in rows}

        self.assertEqual(evidence["openai"], "release-certified")
        self.assertEqual(evidence["meta"], "contract-supported")
        self.assertEqual(evidence["bedrock"], "experimental/native-only")

        markdown = render_provider_support_markdown(rows)
        self.assertIn("| openai | release-certified |", markdown)
        self.assertIn("| meta | contract-supported |", markdown)
        self.assertIn("| bedrock | experimental/native-only |", markdown)

    def test_release_certification_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown provider.*missing"):
            build_provider_support_rows(
                [create_openai(api_key="test")],
                validated_release_certifications={"missing"},
            )

    def test_non_portable_provider_rejects_portable_model_construction(self) -> None:
        provider = create_bedrock(client=_FakeBedrockClient())

        with self.assertRaises(UnsupportedFeatureError):
            provider("anthropic.claude-sonnet-4")

        native_model = provider.native.language_model("anthropic.claude-sonnet-4")
        self.assertEqual(native_model.provider, "bedrock")

    def test_qwen_rejects_native_only_operations_through_portable_namespace(self) -> None:
        provider = create_qwen(api_key="test")

        for operation in (
            provider.grounded_language_model,
            provider.transcription_model,
            provider.speech_model,
        ):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises(UnsupportedFeatureError):
                    operation("test-model")

        self.assertEqual(provider.native.grounded_language_model("test-model").provider, "qwen")
        self.assertEqual(provider.native.transcription_model("test-model").provider, "qwen")
        self.assertEqual(provider.native.speech_model("test-model").provider, "qwen")

    def test_portable_model_factories_match_operation_metadata(self) -> None:
        providers = [
            create_openai(api_key="test"),
            create_azure_openai(api_key="test", endpoint="https://example.openai.azure.com"),
            create_anthropic(api_key="test"),
            create_gemini(api_key="test"),
            create_vertex(access_token="test", project_id="project"),
            create_deepseek(api_key="test"),
            create_qwen(api_key="test"),
            create_kimi(api_key="test"),
            create_vllm(api_key="test"),
            create_meta(api_key="test"),
        ]
        operations = {
            "embeddings": ("embeddings", "embedding_model"),
            "grounding": ("grounding", "grounded_language_model"),
            "transcription": ("transcription", "transcription_model"),
            "speech": ("speech", "speech_model"),
        }

        for provider in providers:
            for label, (flag_name, method_name) in operations.items():
                supported = getattr(provider.portable_support, flag_name)
                method = getattr(provider.portable, method_name)
                with self.subTest(provider=provider.name, operation=label, supported=supported):
                    if supported:
                        self.assertEqual(method("test-model").provider, provider.name)
                    else:
                        with self.assertRaises(UnsupportedFeatureError):
                            method("test-model")


class PortableModelOperationGuardTests(IsolatedAsyncioTestCase):
    async def test_generate_and_stream_enforce_operation_level_portable_metadata(self) -> None:
        provider = create_deepseek(api_key="test")
        provider.portable_support.structured_output = False
        model = provider("deepseek-v4-pro")
        request = ModelGenerateInput(
            structured_output=StructuredOutputConfig(
                schema={"type": "object"},
                mode="native",
            )
        )

        with self.assertRaisesRegex(UnsupportedFeatureError, "structured output"):
            await model.generate(request)
        with self.assertRaisesRegex(UnsupportedFeatureError, "structured output"):
            await model.stream(request)
