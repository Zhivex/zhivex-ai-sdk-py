from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .providers.base import ProviderBundle
from .types import NativeSupport, PortableProviderTier, PortableSupport


@dataclass(slots=True)
class ProviderSupportRow:
    provider: str
    tier: PortableProviderTier
    portable_badge: bool
    portable_support: PortableSupport
    native_support: NativeSupport


def build_provider_support_rows(providers: Mapping[str, ProviderBundle] | Iterable[ProviderBundle]) -> list[ProviderSupportRow]:
    bundles = providers.values() if isinstance(providers, Mapping) else providers
    rows = [
        ProviderSupportRow(
            provider=bundle.name,
            tier=bundle.portable_support.tier,
            portable_badge=bundle.portable_support.portable_badge,
            portable_support=bundle.portable_support,
            native_support=bundle.native_support,
        )
        for bundle in bundles
    ]
    return sorted(rows, key=lambda row: row.provider)


def render_provider_support_markdown(rows: Iterable[ProviderSupportRow]) -> str:
    materialized = list(rows)
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
        "Files",
        "File Search",
        "Images",
        "Uploads",
        "Moderations",
        "Batches",
        "Containers",
        "Skills",
        "Realtime",
        "Responses",
        "Conversations",
    ]

    portable_table = _render_table(
        portable_headers,
        [
            [
                row.provider,
                row.tier,
                _yes_no(row.portable_badge),
                _yes_no(row.portable_support.text_generation),
                _yes_no(row.portable_support.streaming),
                _yes_no(row.portable_support.structured_output),
                _yes_no(row.portable_support.tools),
                _yes_no(row.portable_support.embeddings),
                _yes_no(row.portable_support.grounding),
                _yes_no(row.portable_support.retrieval),
                _yes_no(row.portable_support.transcription),
                _yes_no(row.portable_support.speech),
            ]
            for row in materialized
        ],
    )
    native_table = _render_table(
        native_headers,
        [
            [
                row.provider,
                _yes_no(row.native_support.files),
                _yes_no(row.native_support.file_search),
                _yes_no(row.native_support.images),
                _yes_no(row.native_support.uploads),
                _yes_no(row.native_support.moderations),
                _yes_no(row.native_support.batches),
                _yes_no(row.native_support.containers),
                _yes_no(row.native_support.skills),
                _yes_no(row.native_support.realtime),
                _yes_no(row.native_support.responses),
                _yes_no(row.native_support.conversations),
            ]
            for row in materialized
        ],
    )
    return "\n".join(
        [
            "### Portable Support",
            "",
            portable_table,
            "",
            "### Native Extras",
            "",
            native_table,
        ]
    )


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
