from __future__ import annotations

from pathlib import Path
import re
from typing import Any


def _load_document_class():
    try:
        from docx import Document
    except Exception as error:  # pragma: no cover - exercised in integration tests
        raise RuntimeError('The "docx" skill requires python-docx. Install it with `pip install "zhivex-ai-sdk[docx]"`.') from error
    return Document


def _heading_level(style_name: str) -> int | None:
    match = re.search(r"(\d+)$", style_name.strip())
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def run(payload: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    Document = _load_document_class()
    project_root = Path(context["project_root"])
    input_path = Path(str(payload["input_path"]))
    if not input_path.is_absolute():
        input_path = (project_root / input_path).resolve()

    document = Document(str(input_path))
    outline: list[dict[str, Any]] = []
    paragraphs_payload: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        text_parts.append(text)
        style_name = getattr(getattr(paragraph, "style", None), "name", "") or ""
        paragraphs_payload.append({"text": text, "style": style_name})
        if style_name.lower().startswith("heading"):
            outline.append({"text": text, "style": style_name, "level": _heading_level(style_name)})
    tables: list[dict[str, Any]] = []
    for table_index, table in enumerate(document.tables):
        rows: list[list[str]] = []
        for row in table.rows:
            rows.append([cell.text for cell in row.cells])
        tables.append(
            {
                "index": table_index,
                "row_count": len(rows),
                "column_count": max((len(row) for row in rows), default=0),
                "rows": rows,
            }
        )
    core = getattr(document, "core_properties", None)
    properties = {}
    if core is not None:
        properties = {
            "author": getattr(core, "author", None),
            "category": getattr(core, "category", None),
            "comments": getattr(core, "comments", None),
            "keywords": getattr(core, "keywords", None),
            "language": getattr(core, "language", None),
            "subject": getattr(core, "subject", None),
            "title": getattr(core, "title", None),
        }
    include_paragraphs = bool(payload.get("include_paragraphs", False))
    return {
        "output": {
            "path": str(input_path),
            "outline": outline,
            "heading_count": len(outline),
            "paragraph_count": len(paragraphs_payload),
            "table_count": len(tables),
            "tables": tables,
            "properties": properties,
            "paragraphs": paragraphs_payload if include_paragraphs else [],
            "text": "\n".join(text_parts),
        }
    }
