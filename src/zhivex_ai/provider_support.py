from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .providers.base import ProviderBundle
from .types import AgentCapabilities, NativeSupport, PortableProviderTier, PortableSupport

TIER_1_PROVIDERS = (
    "openai",
    "anthropic",
    "azure-openai",
    "gemini",
    "vertex",
    "qwen",
    "kimi",
    "deepseek",
    "vllm",
)
README_SUPPORT_MATRIX_BEGIN = "<!-- BEGIN GENERATED SUPPORT MATRIX -->"
README_SUPPORT_MATRIX_END = "<!-- END GENERATED SUPPORT MATRIX -->"


@dataclass(slots=True)
class ProviderSupportRow:
    provider: str
    tier: PortableProviderTier
    portable_badge: bool
    portable_support: PortableSupport
    native_support: NativeSupport
    agent_capabilities: AgentCapabilities


def build_provider_support_rows(providers: Mapping[str, ProviderBundle] | Iterable[ProviderBundle]) -> list[ProviderSupportRow]:
    bundles = providers.values() if isinstance(providers, Mapping) else providers
    rows = [
        ProviderSupportRow(
            provider=bundle.name,
            tier=bundle.portable_support.tier,
            portable_badge=bundle.portable_support.portable_badge,
            portable_support=bundle.portable_support,
            native_support=bundle.native_support,
            agent_capabilities=bundle.agent_capabilities,
        )
        for bundle in bundles
    ]
    return sorted(rows, key=lambda row: row.provider)


def get_tier_1_provider_rows(rows: Iterable[ProviderSupportRow]) -> list[ProviderSupportRow]:
    row_map = {row.provider: row for row in rows}
    return [row_map[provider] for provider in TIER_1_PROVIDERS if provider in row_map]


def render_provider_support_markdown(rows: Iterable[ProviderSupportRow]) -> str:
    materialized = list(rows)
    tier_1_rows = get_tier_1_provider_rows(materialized)
    portable_headers = [
        "Provider",
        "Tier",
        "Portable Badge",
        "Text",
        "Streaming",
        "Structured Output",
        "Tools",
        "Embeddings",
        "Grounding",
        "Retrieval",
        "Transcription",
        "Speech",
    ]
    native_headers = [
        "Provider",
        "Text",
        "Streaming",
        "Structured Output",
        "Tools",
        "Embeddings",
        "Grounding",
        "Transcription",
        "Speech",
        "Files",
        "File Search",
        "Images",
        "Uploads",
        "Moderations",
        "Batches",
        "Videos",
        "Media",
        "Interactions",
        "Containers",
        "Skills",
        "Realtime",
        "Responses",
        "Conversations",
        "Caches",
        "Token Count",
        "Formulas",
    ]
    agent_headers = [
        "Provider",
        "Support Tier",
        "Tool Choice None",
        "Approval Requests",
        "Hosted Web Search",
        "Hosted File Search",
        "Remote MCP",
        "Computer Use",
        "Code Execution",
        "Toolsets",
    ]

    portable_table = _render_table(
        portable_headers,
        [
            [
                row.provider,
                row.tier,
                _yes_no(row.portable_badge),
                _portable_yes_no(row, row.portable_support.text_generation),
                _portable_yes_no(row, row.portable_support.streaming),
                _portable_yes_no(row, row.portable_support.structured_output),
                _portable_yes_no(row, row.portable_support.tools),
                _portable_yes_no(row, row.portable_support.embeddings),
                _portable_yes_no(row, row.portable_support.grounding),
                _portable_yes_no(row, row.portable_support.retrieval),
                _portable_yes_no(row, row.portable_support.transcription),
                _portable_yes_no(row, row.portable_support.speech),
            ]
            for row in materialized
        ],
    )
    native_table = _render_table(
        native_headers,
        [
            [
                row.provider,
                _yes_no(row.native_support.text_generation),
                _yes_no(row.native_support.streaming),
                _yes_no(row.native_support.structured_output),
                _yes_no(row.native_support.tools),
                _yes_no(row.native_support.embeddings),
                _yes_no(row.native_support.grounding),
                _yes_no(row.native_support.transcription),
                _yes_no(row.native_support.speech),
                _yes_no(row.native_support.files),
                _yes_no(row.native_support.file_search),
                _yes_no(row.native_support.images),
                _yes_no(row.native_support.uploads),
                _yes_no(row.native_support.moderations),
                _yes_no(row.native_support.batches),
                _yes_no(row.native_support.videos),
                _yes_no(row.native_support.media),
                _yes_no(row.native_support.interactions),
                _yes_no(row.native_support.containers),
                _yes_no(row.native_support.skills),
                _yes_no(row.native_support.realtime),
                _yes_no(row.native_support.responses),
                _yes_no(row.native_support.conversations),
                _yes_no(row.native_support.caches),
                _yes_no(row.native_support.count_tokens),
                _yes_no(row.native_support.formulas),
            ]
            for row in materialized
        ],
    )
    agent_table = _render_table(
        agent_headers,
        [
            [
                row.provider,
                row.agent_capabilities.support_tier,
                _yes_no(row.agent_capabilities.tool_choice_none),
                _yes_no(row.agent_capabilities.approval_requests),
                _yes_no(row.agent_capabilities.hosted_web_search),
                _yes_no(row.agent_capabilities.hosted_file_search),
                _yes_no(row.agent_capabilities.remote_mcp),
                _yes_no(row.agent_capabilities.computer_use),
                _yes_no(row.agent_capabilities.code_execution),
                _yes_no(row.agent_capabilities.toolsets),
            ]
            for row in materialized
        ],
    )
    return "\n".join(
        [
            "### Tier-1 Providers",
            "",
            "These providers back the stable surface for production API work in this SDK today:",
            "",
            *[f"- `{row.provider}`" for row in tier_1_rows],
            "",
            "### Portable Support",
            "",
            portable_table,
            "",
            "### Native Extras",
            "",
            native_table,
            "",
            "### Agent Capabilities",
            "",
            agent_table,
        ]
    )


def render_provider_support_readme_block(rows: Iterable[ProviderSupportRow]) -> str:
    content = render_provider_support_markdown(rows)
    return "\n".join([README_SUPPORT_MATRIX_BEGIN, content, README_SUPPORT_MATRIX_END])


def replace_readme_support_matrix(markdown: str, rows: Iterable[ProviderSupportRow]) -> str:
    begin = README_SUPPORT_MATRIX_BEGIN
    end = README_SUPPORT_MATRIX_END
    if begin not in markdown or end not in markdown:
        raise ValueError("README support-matrix markers are missing.")
    start = markdown.index(begin)
    finish = markdown.index(end) + len(end)
    return markdown[:start] + render_provider_support_readme_block(rows) + markdown[finish:]


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _portable_yes_no(row: ProviderSupportRow, value: bool) -> str:
    return _yes_no(value) if row.portable_badge else "N/A"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
