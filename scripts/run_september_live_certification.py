"""Bounded September refresh probes against an explicitly installed wheel.

Writes sanitized, incremental integration evidence; never promotes release policy.
Run with ZHIVEX_LIVE_WHEEL and ZHIVEX_LIVE_EVIDENCE set. No credentials are saved.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

# Import before exposing repo scripts so source cannot shadow the installed wheel.
import zhivex_ai as sdk
from zhivex_ai.types import RetryOptions, RealtimeConnectOptions
from zhivex_ai.experimental import (
    RealtimeAudioOutputEvent,
    RealtimeResponseCompletedEvent,
)
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
if ROOT in Path(sdk.__file__).resolve().parents:
    raise SystemExit("An installed wheel outside the checkout is required.")
sys.path.insert(0, str(ROOT))
os.environ["ZHIVEX_SMOKE_USE_INSTALLED"] = "1"
from scripts.run_live_smoke import (  # noqa: E402
    _run_portable_certification,
    _run_agent_tool_smoke,
    _failure_diagnostic_code,
    _is_external_blocker,
    _safe_error_message,
)

OPTIONS = RetryOptions(timeout_ms=60_000, max_retries=0)
WHEEL = Path(os.environ["ZHIVEX_LIVE_WHEEL"])
OUTPUT = Path(os.environ["ZHIVEX_LIVE_EVIDENCE"])
if OUTPUT.exists():
    raise SystemExit("Evidence already exists; choose a fresh output path.")
MANIFEST = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((ROOT / "src").rglob("*.py"))
}
import zipfile  # noqa: E402

with zipfile.ZipFile(WHEEL) as archive:
    for relative, digest in MANIFEST.items():
        member = relative.removeprefix("src/")
        installed = Path(sdk.__file__).parent.parent / member
        if (
            hashlib.sha256(archive.read(member)).hexdigest() != digest
            or hashlib.sha256(installed.read_bytes()).hexdigest() != digest
        ):
            raise SystemExit(
                "Source, wheel, and installation must match before live probes."
            )
REPORT = {
    "schema_version": 1,
    "kind": "september-refresh-live-integration",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "artifact": {
        "filename": WHEEL.name,
        "sha256": hashlib.sha256(WHEEL.read_bytes()).hexdigest(),
        "installed_module": str(Path(sdk.__file__).resolve()),
        "source_parent": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        ),
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(MANIFEST, sort_keys=True).encode()
        ).hexdigest(),
        "source_manifest": MANIFEST,
    },
    "release_certified": False,
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "results": [],
}
LIMIT = asyncio.Semaphore(3)


def save():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(REPORT, indent=2) + "\n")


async def probe(provider, model, operation, credential, run):
    selected = os.getenv("ZHIVEX_LIVE_ONLY", "")
    if selected and f"{provider}:{model}:{operation}" not in selected.split(","):
        return
    async with LIMIT:
        row = {
            "provider": provider,
            "model": model,
            "operation": operation,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        if not os.getenv(credential):
            row.update(status="blocked", diagnostic="MISSING_CREDENTIAL")
        else:
            try:
                async with asyncio.timeout(150):
                    details = await run()
                row.update(status="passed", details=details or {})
            except Exception as error:
                code = _failure_diagnostic_code(provider, error)
                transport_blocked = isinstance(error, TimeoutError) or type(
                    error
                ).__name__ in {
                    "ReadTimeout",
                    "ConnectTimeout",
                    "ConnectError",
                    "NetworkError",
                }
                if transport_blocked:
                    code = "PROVIDER_TRANSPORT_TIMEOUT_OR_UNAVAILABLE"
                row.update(
                    status="blocked"
                    if _is_external_blocker(code) or transport_blocked
                    else "failed",
                    diagnostic=code,
                    error_type=type(error).__name__,
                    diagnostic_summary=_safe_error_message(error)[:600],
                )
                status = getattr(error, "status", None)
                if isinstance(status, int):
                    row["http_status"] = status
                # No messages/bodies/URLs/headers from external services in artifacts.
        row["finished_at"] = datetime.now(timezone.utc).isoformat()
        REPORT["results"].append(row)
        save()
        print(
            f"{provider} {model} {operation}: {row['status']} {row.get('diagnostic', '')}",
            flush=True,
        )


async def portable(provider, model, effort, operation):
    p = {
        "deepseek": sdk.create_deepseek,
        "meta": sdk.create_meta,
        "openai": sdk.create_openai,
        "anthropic": sdk.create_anthropic,
        "gemini": sdk.create_gemini,
        "qwen": sdk.create_qwen,
    }[provider]()
    m = p(model)
    reasoning = sdk.ReasoningConfig(effort=effort) if effort else None
    if operation == "generation":
        out = await sdk.generate_text(
            model=m,
            prompt="Reply with exactly LIVE_OK.",
            reasoning=reasoning,
            max_tokens=1024,
            timeout_ms=45_000,
            max_retries=0,
        )
        assert out.text.strip().rstrip(".") == "LIVE_OK", "Unexpected generation output"
    elif operation == "streaming-and-structured-output":
        completed = set()
        await _run_portable_certification(
            provider=provider,
            model=m,
            reasoning=reasoning,
            structured_max_tokens=None if provider == "qwen" else 1024,
            completed_operations=completed,
        )
        return {"completed": sorted(completed)}
    else:
        await _run_agent_tool_smoke(provider=provider, model=m)


async def vision(model):
    # A locally generated, solid-red PNG: synthetic data only.
    import struct
    import zlib

    def chunk(kind, data):
        return (
            struct.pack("!I", len(data))
            + kind
            + data
            + struct.pack("!I", zlib.crc32(kind + data))
        )

    png = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack("!2I5B", 64, 64, 8, 2, 0, 0, 0)
    )
    png += chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 64) * 64)) + chunk(
        b"IEND", b""
    )
    out = await sdk.generate_text(
        model=sdk.create_deepseek().native.language_model(model),
        messages=[
            sdk.ModelMessage(
                role="user",
                parts=[
                    sdk.TextPart(
                        text="What single color fills this image? Reply one word."
                    ),
                    sdk.ImagePart(
                        image=base64.b64encode(png).decode(), media_type="image/png"
                    ),
                ],
            )
        ],
        reasoning=sdk.ReasoningConfig(effort="low"),
        provider_options={"top_p": 0.98},
        max_tokens=1024,
        timeout_ms=45_000,
        max_retries=0,
    )
    assert "red" in out.text.lower(), "Vision color check failed"


async def compact():
    client = sdk.create_anthropic().native.messages()
    body = {
        "model": "claude-fable-5-1",
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": "Remember the project codename is AMBER. " * 100,
            }
        ],
    }
    result = await client.compact(body, OPTIONS)
    blocks = result.get("content", [])
    assert any(
        b.get("type") == "compaction" and b.get("content") and b.get("signature")
        for b in blocks
    ), "No usable signed block"
    response = await client.create(
        {
            **body,
            "messages": [
                {"role": "assistant", "content": blocks},
                {
                    "role": "user",
                    "content": "What is the project codename? Reply one word.",
                },
            ],
        },
        OPTIONS,
    )
    assert any("AMBER" in b.get("text", "") for b in response.get("content", [])), (
        "Compaction replay lost context"
    )
    return {"signed_block": True, "replay": True}


async def image(model):
    out = (
        await sdk.create_openai()
        .native.images()
        .generate(
            model=model,
            prompt="A solid blue square. No text.",
            quality="max",
            size="1024x1024",
        )
    )
    assert out.images, "No image returned"
    return {"images": len(out.images)}


async def music():
    out = (
        await sdk.create_gemini()
        .native.media()
        .generate_music(
            model="lyria-3.5",
            prompt="A short calm solo piano instrumental.",
            options=OPTIONS,
        )
    )
    assert out.media and out.media[0].media_type.startswith("audio/"), (
        "No audio returned"
    )
    return {"audio_mime": out.media[0].media_type}


async def gemini_live(model):
    opts = (
        {"generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}}}
        if model.endswith("thinking")
        else {}
    )
    session = (
        await sdk.create_gemini()
        .native.realtime_model(model)
        .connect(
            sdk.RealtimeSessionConfig(
                instructions="Keep answers short.", provider_options=opts
            ),
            RealtimeConnectOptions(timeout_ms=30_000),
        )
    )
    try:
        await session.send_text("Say hello in one word.")
        count = 0
        async for event in session.event_stream():
            if isinstance(event, RealtimeAudioOutputEvent):
                count += len(event.audio)
            if isinstance(event, RealtimeResponseCompletedEvent):
                assert count > 0, "No audio before completion"
                return {"audio_bytes": count, "turn_completed": True}
        raise AssertionError("Stream ended without turn completion")
    finally:
        await session.aclose()


async def openai_live():
    session = (
        await sdk.create_openai()
        .native.live()
        .connect(
            {
                "model": "gpt-live-1",
                "audio": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "output": {"voice": "marin"},
                },
                "delegation": {"type": "client"},
            },
            RealtimeConnectOptions(timeout_ms=30_000),
        )
    )
    try:
        final = await session.finish(timeout_ms=15_000)
        assert final.get("type") == "session.closed"
        return {
            "startup": True,
            "finalization": True,
            "usage_present": "usage" in final,
        }
    finally:
        await session.aclose()


async def agents():
    client = sdk.create_openai().native.agent_sessions()
    existing = os.getenv("ZHIVEX_LIVE_AGENT_SESSION")
    if existing:
        created = await client.retrieve(existing, OPTIONS)
    else:
        created = await client.create(
            {
                "agent": {"model": "gpt-6-astra", "instructions": "Reply briefly."},
                "environment": {"type": "openai_hosted"},
            },
            OPTIONS,
        )
    sid = created["id"]
    resources = REPORT.setdefault("temporary_resources", [])
    resource = {
        "provider": "openai",
        "kind": "agent-session",
        "id": sid,
        "deleted": False,
    }
    resources.append(resource)
    save()
    task = None
    try:
        assert (await client.retrieve(sid, OPTIONS))["id"] == sid
        await client.list(limit=1, options=OPTIONS)
        await client.items(sid, options=OPTIONS)
        try:
            await client.cancel(sid, OPTIONS)
        except sdk.ProviderHTTPError as error:
            # Idle sessions have no active turn to cancel.
            if error.status != 409:
                raise
        from contextlib import aclosing

        ready = asyncio.Event()
        original_fetch = client.fetch

        async def fetch(url, **kwargs):
            response = await original_fetch(url, **kwargs)
            if kwargs.get("stream"):
                ready.set()
            return response

        client.fetch = fetch

        async def consume():
            async with aclosing(client.events(sid, OPTIONS)) as events:
                async for event in events:
                    kind = event.get("type")
                    if kind in {
                        "error",
                        "agent.session.failed",
                        "agent.session.environment.failed",
                        "agent.session.turn.failed",
                    }:
                        raise AssertionError("Managed agent emitted failure event")
                    if (
                        kind == "agent.session.turn.completed"
                        and (event.get("turn") or {}).get("subagent_id") is None
                    ):
                        return
            raise AssertionError("No completed root turn")

        task = asyncio.create_task(consume())
        await asyncio.wait_for(ready.wait(), 20)
        await client.send_events(
            sid,
            [
                {
                    "type": "agent.session.input.message",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "Reply exactly LIVE_OK. Do not use tools.",
                                }
                            ],
                        }
                    ],
                }
            ],
            OPTIONS,
        )
        await asyncio.wait_for(task, 60)
        items = await client.items(sid, options=OPTIONS)
        assert items.get("data"), "No persisted session items"
        return {
            "create": not bool(existing),
            "resumed_smoke_session": bool(existing),
            "retrieve": True,
            "list": True,
            "items": True,
            "idle_cancel_contract": True,
            "send_events": True,
            "sse_completed_turn": True,
        }
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for attempt in range(4):
            try:
                await client.delete(sid, OPTIONS)
                break
            except sdk.ProviderHTTPError as error:
                if error.status != 409 or attempt == 3:
                    raise
                await asyncio.sleep(5)
        resource["deleted"] = True
        save()


async def cache():
    client = sdk.create_openai().native.responses()
    body = {
        "model": "gpt-6-astra",
        "input": "Reply OK.",
        "max_output_tokens": 1024,
        "reasoning": {"effort": "low"},
        "instructions": "Use concise answers. " * 400,
    }
    first = await client.create(body, OPTIONS)
    second = await client.create(
        {**body, "prompt_cache_options": {"comparison_response_id": first["id"]}},
        OPTIONS,
    )
    assert second.get("prompt_cache_diagnostics") is not None, "Missing diagnostics"
    return {"diagnostics_present": True}


async def main():
    jobs = []
    targets = [
        ("deepseek", "deepseek-flash", "none", "DEEPSEEK_API_KEY"),
        ("meta", "muse-spark-1.3", "low", "MODEL_API_KEY"),
        ("openai", "gpt-6-astra", "low", "OPENAI_API_KEY"),
        ("anthropic", "claude-fable-5-1", None, "ANTHROPIC_API_KEY"),
        ("gemini", "gemini-3.8-flash", "low", "GOOGLE_API_KEY"),
        ("qwen", "qwen3.8-max-0902", "none", "QWEN_API_KEY"),
    ]
    for p, m, e, k in targets:
        for op in ("generation", "streaming-and-structured-output", "agent-tool"):
            jobs.append(
                probe(p, m, op, k, lambda p=p, m=m, e=e, op=op: portable(p, m, e, op))
            )
    for m in ("deepseek-flash", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"):
        jobs.append(
            probe(
                "deepseek",
                m,
                "vision-reasoning-top-p",
                "DEEPSEEK_API_KEY",
                lambda m=m: vision(m),
            )
        )
    jobs.append(
        probe(
            "anthropic",
            "claude-fable-5-1",
            "signed-compaction-replay",
            "ANTHROPIC_API_KEY",
            compact,
        )
    )
    for m in ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare"):
        jobs.append(
            probe(
                "openai",
                m,
                "image-generation-max",
                "OPENAI_API_KEY",
                lambda m=m: image(m),
            )
        )
    jobs.append(probe("gemini", "lyria-3.5", "music", "GOOGLE_API_KEY", music))
    for m in ("gemini-3.8-live", "gemini-3.8-live-extended-thinking"):
        jobs.append(
            probe(
                "gemini",
                m,
                "live-audio-turn",
                "GOOGLE_API_KEY",
                lambda m=m: gemini_live(m),
            )
        )
    jobs.append(
        probe("openai", "gpt-live-1", "live-start-close", "OPENAI_API_KEY", openai_live)
    )
    jobs.append(
        probe(
            "openai",
            "gpt-6-astra",
            "managed-agents-lifecycle",
            "OPENAI_API_KEY",
            agents,
        )
    )
    jobs.append(
        probe("openai", "gpt-6-astra", "cache-diagnostics", "OPENAI_API_KEY", cache)
    )
    save()
    await asyncio.gather(*jobs)
    await sdk.aclose_default_clients()
    REPORT["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
    return int(any(r["status"] != "passed" for r in REPORT["results"]))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
