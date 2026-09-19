"""Bounded native Mistral rawPredict checks from an exact installed wheel."""
from __future__ import annotations

import argparse
import asyncio
import ast
import base64
from contextlib import aclosing
from datetime import datetime, timezone
import json
from pathlib import Path

import google.auth
from zhivex_ai import create_vertex
from zhivex_ai.types import RetryOptions
from zhivex_ai._sse import parse_sse

if __package__:
    from .verify_vertex_integration import verify_wheel
else:
    from verify_vertex_integration import verify_wheel


def validate_fim(text, finish):
    if finish != "stop":
        raise RuntimeError("FIM did not complete")
    expression = ast.parse("42 + " + text.strip(), mode="eval").body
    if not (isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add)
            and isinstance(expression.left, ast.Constant) and expression.left.value == 42
            and isinstance(expression.right, ast.Constant) and type(expression.right.value) is int
            and expression.right.value == 58):
        raise RuntimeError("FIM arithmetic completion mismatch")


def validate_ocr(response):
    pages = response.get("pages", [])
    if not pages or not all(isinstance(page, dict) and isinstance(page.get("markdown"), str) for page in pages):
        raise RuntimeError("OCR pages missing")
    text = "\n".join(page["markdown"] for page in pages)
    if "ORCHID-7284" not in text or "turquoise" not in text.lower():
        raise RuntimeError("OCR document marker mismatch")


async def run(args):
    if args.kind == "fim" and args.model != "codestral-2":
        raise SystemExit("FIM requires codestral-2")
    if args.kind == "ocr" and args.model != "mistral-ocr-2505":
        raise SystemExit("OCR requires mistral-ocr-2505")
    digest = verify_wheel(args.wheel)
    if args.output.exists():
        raise SystemExit("Choose a fresh evidence path.")
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=args.project)
    garden = create_vertex(credentials=credentials, project_id=args.project, location=args.location).native.model_garden()
    report = {"schema_version": 1, "provider": "vertex", "model": args.model,
              "location": args.location, "mode": "native-mistral-raw-predict",
              "recorded_at": datetime.now(timezone.utc).isoformat(),
              "wheel_sha256": digest, "evidence_status": "integration-only",
              "operations": {}, "diagnostics": {}}
    for stream in ((False,) if args.kind == "ocr" else (False, True)):
        operation = "ocr-document" if args.kind == "ocr" else (("fim-streaming" if stream else "fim-generation") if args.kind == "fim" else ("streaming" if stream else "generation"))
        try:
            async with asyncio.timeout(75):
                options = RetryOptions(timeout_ms=60000, max_retries=0)
                if args.kind == "ocr":
                    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/vertex/document-marker.pdf"
                    response = await garden.mistral_ocr({"document": {"type": "document_url", "document_url": "data:application/pdf;base64," + base64.b64encode(fixture.read_bytes()).decode()}, "pages": [0]}, model=args.model, options=options)
                    validate_ocr(response)
                elif args.kind == "fim":
                    response = await garden.codestral_fim({"prompt": "42 + ", "suffix": " = 100", "max_tokens": 32, "stream": stream, "temperature": 0}, model=args.model, options=options)
                else:
                    response = await garden.raw_predict(
                        publisher="mistralai", model=args.model, stream=stream,
                        body={"model": args.model.split("@")[0], "stream": stream,
                              "messages": [{"role": "user", "content": "Reply with exactly VERTEX_MISTRAL_OK."}],
                              "max_tokens": 128}, options=options,
                    )
                if args.kind == "ocr":
                    text, finish = "", None
                elif stream:
                    text, finish = "", None
                    async with aclosing(response.iter_lines()) as lines:
                        async for event in parse_sse(lines):
                            if event.data == "[DONE]":
                                break
                            payload = json.loads(event.data)
                            if payload.get("error"):
                                raise RuntimeError("Upstream stream error")
                            for choice in payload.get("choices", []):
                                if choice.get("index", 0) == 0:
                                    text += (choice.get("delta") or {}).get("content") or ""
                                    finish = choice.get("finish_reason") or finish
                else:
                    choice = response["choices"][0]
                    text, finish = choice["message"]["content"], choice.get("finish_reason")
                if args.kind == "fim":
                    validate_fim(text, finish)
                elif args.kind == "chat" and ("VERTEX_MISTRAL_OK" not in text or finish != "stop"):
                    raise RuntimeError("Marker or terminal reason mismatch")
                report["operations"][operation] = "passed"
        except Exception as error:
            report["operations"][operation] = "failed"
            report["diagnostics"][operation] = {"error_type": type(error).__name__, "http_status": getattr(error, "status", None)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(operation, report["operations"][operation], flush=True)
    return int(any(value != "passed" for value in report["operations"].values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", choices=("us-central1", "europe-west4"), default="us-central1")
    parser.add_argument("--model", choices=("mistral-medium-3", "mistral-small-2503", "codestral-2", "mistral-ocr-2505"), default="mistral-medium-3")
    parser.add_argument("--kind", choices=("chat", "fim", "ocr"), default="chat")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
