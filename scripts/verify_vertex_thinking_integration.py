"""Native Gemma/GLM thinking toggle checks, without persisting reasoning text."""
from __future__ import annotations

from contextlib import aclosing
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path

import google.auth
from zhivex_ai import create_vertex, create_text_message
from zhivex_ai.types import ModelGenerateInput, StreamTextDeltaEvent, StreamProviderDataEvent, StreamFinishEvent
from verify_vertex_integration import verify_wheel


async def run(args):
    digest = verify_wheel(args.wheel)
    if args.output.exists():
        raise SystemExit("Choose a fresh evidence path.")
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=args.project)
    model = create_vertex(credentials=credentials, project_id=args.project, location="global").native.model_garden().language_model(args.model)
    report = {"schema_version": 1, "provider": "vertex", "model": args.model, "location": "global",
              "scope": "native-thinking-toggle-stream-only" if args.stream_only else "native-thinking-toggle", "wheel_sha256": digest, "evidence_status": "integration-only",
              "recorded_at": datetime.now(timezone.utc).isoformat(), "operations": {}, "diagnostics": {}}
    for enabled in (False, True):
        for streaming in ((True,) if args.stream_only else (False, True)):
            name = ("stream" if streaming else "generate") + ("-thinking-on" if enabled else "-thinking-off")
            request = ModelGenerateInput(messages=[create_text_message("user", "Calculate 17 times 19. Reply with only the decimal integer answer.")], max_tokens=1024, max_retries=0, timeout_ms=60000, provider_options={"chat_template_kwargs": {"enable_thinking": enabled}})
            try:
                async with asyncio.timeout(75):
                    if streaming:
                        text, reasoning_chars, finishes = "", 0, []
                        async with aclosing(await model.stream(request)) as events:
                            async for event in events:
                                if isinstance(event, StreamTextDeltaEvent):
                                    text += event.text_delta
                                elif isinstance(event, StreamProviderDataEvent):
                                    reasoning_chars += len(event.data.get("reasoning_content", ""))
                                elif isinstance(event, StreamFinishEvent):
                                    finishes.append(event.finish_reason)
                        finish = finishes[-1] if finishes else None
                    else:
                        result = await model.generate(request)
                        text, finish = result.text, result.finish_reason
                        reasoning_chars = len((result.raw_response or {}).get("choices", [{}])[0].get("message", {}).get("reasoning_content") or "")
                    report["diagnostics"][name] = {"finish_reason": finish, "text_chars": len(text), "reasoning_chars": reasoning_chars, "marker_matched": "323" in text}
                    if "323" not in text or finish != "stop" or bool(reasoning_chars) != enabled:
                        raise RuntimeError("Thinking toggle or final answer check failed")
                report["operations"][name] = "passed"
            except Exception as error:
                report["operations"][name] = "failed"
                report["diagnostics"].setdefault(name, {}).update({"error_type": type(error).__name__, "http_status": getattr(error, "status", None)})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(name, report["operations"][name], flush=True)
    return int(any(value != "passed" for value in report["operations"].values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream-only", action="store_true", help="Check only streaming; does not certify generation")
    parser.add_argument("--model", choices=["google/gemma-4-26b-a4b-it-maas", "zai-org/glm-5.2-maas"], default="google/gemma-4-26b-a4b-it-maas")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
