from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_document_class():
    try:
        from docx import Document
    except Exception as error:  # pragma: no cover - exercised in integration tests
        raise RuntimeError('The "docx" skill requires python-docx. Install it with `pip install "zhivex-ai-sdk[docx]"`.') from error
    return Document


def _apply_properties(document: Any, payload: dict[str, Any]) -> None:
    properties = payload.get("properties")
    if not isinstance(properties, dict):
        return
    core = getattr(document, "core_properties", None)
    if core is None:
        return
    for field in ("author", "category", "comments", "keywords", "language", "subject", "title"):
        value = properties.get(field)
        if value is None:
            continue
        setattr(core, field, str(value))


def _add_paragraph_block(document: Any, value: Any, *, style: str | None = None) -> None:
    if isinstance(value, dict):
        text = str(value.get("text") or "").strip()
        if not text:
            return
        paragraph_style = str(value.get("style") or style or "").strip() or None
        paragraph = document.add_paragraph(text)
        if paragraph_style:
            paragraph.style = paragraph_style
        return
    text = str(value).strip()
    if text:
        paragraph = document.add_paragraph(text)
        if style:
            paragraph.style = style


def _add_list(document: Any, items: list[Any], *, ordered: bool) -> None:
    style = "List Number" if ordered else "List Bullet"
    for item in items:
        _add_paragraph_block(document, item, style=style)


def _add_sections(document: Any, sections: list[Any]) -> None:
    for section in sections:
        if not isinstance(section, dict):
            continue
        heading = str(section.get("heading") or "").strip()
        level = int(section.get("level") or 1)
        if heading:
            document.add_heading(heading, level=max(1, min(level, 9)))
        body = section.get("body")
        if body:
            _add_paragraph_block(document, body)
        for paragraph in list(section.get("paragraphs") or []):
            _add_paragraph_block(document, paragraph)
        bullet_list = list(section.get("bullet_list") or [])
        if bullet_list:
            _add_list(document, bullet_list, ordered=False)
        numbered_list = list(section.get("numbered_list") or [])
        if numbered_list:
            _add_list(document, numbered_list, ordered=True)


def _add_tables(document: Any, tables: list[Any]) -> None:
    for table_payload in tables:
        if not isinstance(table_payload, dict):
            continue
        title = str(table_payload.get("title") or "").strip()
        if title:
            document.add_paragraph(title, style="Intense Quote")
        rows = list(table_payload.get("rows") or [])
        if not rows:
            continue
        width = max(len(row) if isinstance(row, list) else 0 for row in rows)
        table = document.add_table(rows=len(rows), cols=max(width, 1))
        style_name = str(table_payload.get("style") or "").strip()
        if style_name:
            table.style = style_name
        for row_index, row in enumerate(rows):
            if not isinstance(row, list):
                continue
            for column_index, value in enumerate(row):
                table.cell(row_index, column_index).text = str(value)


def run(payload: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    Document = _load_document_class()
    project_root = Path(context["project_root"])
    output_path = Path(str(payload["output_path"]))
    if not output_path.is_absolute():
        output_path = (project_root / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = Document()
    _apply_properties(document, payload)
    title = str(payload.get("title") or "").strip()
    if title:
        document.add_heading(title, level=0)
    subtitle = str(payload.get("subtitle") or "").strip()
    if subtitle:
        document.add_paragraph(subtitle, style="Subtitle")

    for paragraph in list(payload.get("paragraphs") or []):
        _add_paragraph_block(document, paragraph)

    _add_sections(document, list(payload.get("sections") or []))
    bullet_list = list(payload.get("bullet_list") or [])
    if bullet_list:
        _add_list(document, bullet_list, ordered=False)
    numbered_list = list(payload.get("numbered_list") or [])
    if numbered_list:
        _add_list(document, numbered_list, ordered=True)

    _add_tables(document, list(payload.get("tables") or []))

    document.save(output_path)
    return {
        "output": {
            "path": str(output_path),
            "title": title or None,
            "subtitle": subtitle or None,
            "paragraph_count": len(document.paragraphs),
            "section_count": len(list(payload.get("sections") or [])),
            "table_count": len(list(payload.get("tables") or [])),
        },
        "artifacts": [
            {
                "name": output_path.name,
                "path": str(output_path),
                "media_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "role": "primary",
            }
        ],
    }
