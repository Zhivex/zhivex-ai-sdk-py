---
name: docx
description: Use this skill whenever the user wants to create, read, edit, or analyze Word documents (.docx files).
---

# DOCX Skill

Use this skill whenever a Word document is the primary input, output, or working format.

This skill is for real `.docx` workflows, not just nicely formatted text. Prefer it when the user wants a document they can open in Microsoft Word or another `.docx`-compatible editor.

## When To Use It

Activate this skill when the user asks to:

- create a report, memo, brief, letter, proposal, template, or other deliverable as a Word document
- edit or revise an existing `.docx` file
- append sections, replace text, or add tables and lists in a `.docx`
- inspect a `.docx` and extract headings, paragraphs, tables, metadata, or plain text
- convert structured content into a polished editable document

Strong indicators include mentions of:

- `Word`
- `.docx`
- `document`
- `report`
- `memo`
- `letter`
- `template`
- `editable deliverable`

## Core Behavior

When this skill is active:

1. Decide whether the task is primarily `create`, `edit`, or `analyze`.
2. Prefer structured document operations over freeform prose generation.
3. Preserve the fact that the output is a file artifact, not just chat text.
4. Ask for missing document structure only when it is necessary to produce a useful file.

## Entrypoint Selection

Use `create` when:

- the user wants a new `.docx`
- the document should be assembled from structured content
- the user asks for a polished deliverable from scratch

Use `edit` when:

- there is an existing `.docx`
- the user wants to append content, replace text, or revise an existing document
- preserving the current document is more important than recreating it

Use `analyze` when:

- the user wants to inspect or summarize an existing `.docx`
- the goal is to extract headings, tables, paragraph structure, or metadata
- no document mutation is requested

## Authoring Guidelines

- Prefer clear structure: title, subtitle when useful, sections, paragraphs, and tables.
- Use lists when the user provides bullets, steps, or grouped items.
- Preserve user wording where fidelity matters, especially for formal or client-facing content.
- When details are underspecified, choose a professional default layout instead of blocking.
- Keep edits additive and conservative unless the user clearly asks for a rewrite.
- If the user provides business metadata such as author or subject, include it through document properties when possible.

## Editing Guidelines

- Prefer targeted replacements over rewriting the entire document.
- When appending sections, maintain the existing document tone and structure.
- If replacing text, avoid broad unintended substitutions; keep replacements narrow and literal.
- If an edit could materially overwrite prior content, preserve the original unless the user explicitly wants in-place modification.

## Analysis Guidelines

- Return structured findings, not just a text dump.
- Capture headings, table counts, paragraph counts, and important metadata when available.
- Include plain text in addition to structure so the caller can summarize or transform the content downstream.

## Output Expectations

When creating or editing a document:

- produce a real `.docx` artifact
- return concise structured metadata about what was written
- prefer stable, editable formatting over ornate layout tricks

When analyzing a document:

- return structure first
- include extracted text
- include tables and metadata when available

## Examples

- "Create a Word memo with a title page and two sections."
- "Update this `.docx` and append a next-steps section."
- "Replace the old company name throughout this Word file."
- "Analyze this report and tell me its outline and tables."
