"""Exact-wheel Vertex Cloud lifecycle checks with temporary synthetic resources.

Creates only its own short-lived cache and empty agent, and deletes both in
finally blocks. Does not change IAM, existing agents, or existing user data.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from typing import Any

import google.auth
from zhivex_ai import create_vertex, generate_text
from zhivex_ai.types import RetryOptions
if __package__:
    from .verify_vertex_integration import verify_wheel
else:
    from verify_vertex_integration import verify_wheel


async def run(args: argparse.Namespace) -> int:
    digest = verify_wheel(args.wheel)
    if args.output.exists():
        raise SystemExit("Choose a fresh evidence path.")
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
        quota_project_id=args.project,
    )
    provider = create_vertex(
        credentials=credentials, project_id=args.project, location=args.location
    )
    platform = provider.native.agent_platform()
    options = RetryOptions(timeout_ms=30000, max_retries=0)
    operations: dict[str, str] = {}
    diagnostics: dict[str, Any] = {}
    report = {
        "schema_version": 1,
        "provider": "vertex",
        "model": args.model,
        "location": args.location,
        "mode": "standard-adc",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "wheel_sha256": digest,
        "evidence_status": "integration-only",
        "operations": operations,
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    async def check(name: str, call: Any) -> Any:
        try:
            async with asyncio.timeout(90):
                result = await call()
            operations[name] = "passed"
            return result
        except Exception as error:
            operations[name] = "failed"
            diagnostics[name] = {
                "error_type": type(error).__name__,
                "http_status": getattr(error, "status", None),
            }
            raise
        finally:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{name}: {operations[name]}", flush=True)

    async def completed(operation: dict[str, Any]) -> dict[str, Any]:
        if not operation.get("done"):
            operation = await platform.wait_operation(
                operation["name"], timeout_ms=60000
            )
        if operation.get("error"):
            raise RuntimeError("Cloud operation failed")
        return operation.get("response", {})

    cache = None
    try:
        cache = await check(
            "cache-create",
            lambda: provider.caches().create(
                {
                    "model": args.model,
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": "Synthetic SDK cache verification marker VERTEX_CACHE_OK. "
                                    * 1024
                                }
                            ],
                        }
                    ],
                    "ttl": "180s",
                },
                options,
            ),
        )
        await check("cache-get", lambda: provider.caches().get(cache.name, options))
        await check(
            "cache-update",
            lambda: provider.caches().update(cache.name, {"ttl": "240s"}, options),
        )

        async def cache_use() -> None:
            result = await generate_text(
                model=provider.native.language_model(args.model),
                prompt="Return only the verification marker.",
                provider_options={"cachedContent": cache.name},
                max_tokens=512,
                timeout_ms=30000,
                max_retries=0,
            )
            if "VERTEX_CACHE_OK" not in result.text:
                raise RuntimeError("Cache verification marker mismatch")
            cached_tokens = result.steps[-1].response.raw_response.get("usageMetadata", {}).get("cachedContentTokenCount", 0)
            diagnostics["cached_content_tokens"] = cached_tokens
            if not isinstance(cached_tokens, int) or cached_tokens <= 0:
                raise RuntimeError("Response did not confirm cached token use")

        await check("cache-generation", cache_use)
    except Exception:
        pass  # Evidence already records the failing operation; cleanup still runs.
    finally:
        if cache is not None:
            try:
                await check(
                    "cache-delete",
                    lambda: provider.caches().delete(cache.name, options),
                )

                async def absent() -> None:
                    try:
                        await provider.caches().get(cache.name, options)
                    except Exception as error:
                        if getattr(error, "status", None) == 404:
                            return
                        raise
                    raise RuntimeError("Deleted cache remains readable")

                await check("cache-absent", absent)
            except Exception:
                pass

    if args.cache_only:
        required = ("cache-create", "cache-get", "cache-update", "cache-generation", "cache-delete", "cache-absent")
        return int(any(operations.get(name) != "passed" for name in required))

    agent_name = None
    try:

        async def create_agent() -> dict[str, Any]:
            nonlocal agent_name
            operation = await platform.agents().create(
                {"displayName": "zhivex-sdk-smoke-" + uuid.uuid4().hex[:8]},
                options=options,
            )
            # Capture the newly created resource before polling so a timeout
            # still enters cleanup for our own agent.
            operation_name = operation.get("name", "")
            if (
                "/reasoningEngines/" in operation_name
                and "/operations/" in operation_name
            ):
                agent_name = operation_name.rsplit("/operations/", 1)[0]
            return await completed(operation)

        agent = await check("agent-create", create_agent)
        agent_name = agent["name"]

        async def create_session() -> dict[str, Any]:
            return await completed(
                await platform.sessions(agent_name).create(
                    {"userId": "zhivex-synthetic-user"}, options=options
                )
            )

        session = await check("session-create", create_session)
        await check(
            "session-get",
            lambda: platform.sessions(agent_name).get(session["name"], options),
        )
        await check(
            "session-append-event",
            lambda: platform.sessions(agent_name).append_event(
                session["name"],
                {
                    "author": "user",
                    "invocationId": "smoke",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "content": {
                        "role": "user",
                        "parts": [{"text": "Synthetic SDK test event."}],
                    },
                },
                options,
            ),
        )
        await check(
            "session-events",
            lambda: platform.sessions(agent_name).events(
                session["name"], options=options
            ),
        )

        async def create_memory() -> dict[str, Any]:
            return await completed(
                await platform.memory_bank(agent_name).create(
                    {
                        "fact": "Synthetic SDK test preference is blue.",
                        "scope": {"user_id": "zhivex-synthetic-user"},
                    },
                    options=options,
                )
            )

        memory = await check("memory-create", create_memory)
        await check(
            "memory-get",
            lambda: platform.memory_bank(agent_name).get(memory["name"], options),
        )
        await check(
            "memory-list",
            lambda: platform.memory_bank(agent_name).list(options=options),
        )
        await check(
            "memory-retrieve",
            lambda: platform.memory_bank(agent_name).retrieve(
                {
                    "scope": {"user_id": "zhivex-synthetic-user"},
                    "similaritySearchParams": {"searchQuery": "preferred color"},
                },
                options,
            ),
        )
        await check(
            "memory-delete",
            lambda: platform.memory_bank(agent_name).delete(
                memory["name"], options=options
            ),
        )
        await check(
            "session-delete",
            lambda: platform.sessions(agent_name).delete(
                session["name"], options=options
            ),
        )
    except Exception:
        pass
    finally:
        if agent_name:

            async def delete_agent() -> dict[str, Any]:
                return await completed(
                    await platform.agents().delete(
                        agent_name, force=True, options=options
                    )
                )

            try:
                await check("agent-delete", delete_agent)
            except Exception:
                pass

    async def chat() -> None:
        result = await provider.native.model_garden().chat_completions(
            {
                "model": f"google/{args.model}",
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 512,
            },
            options=options,
        )
        if not result.get("choices"):
            raise RuntimeError("Missing chat choices")

    reads = [
        ("batches-list", lambda: provider.batches().list(limit=1, options=options)),
        (
            "model-garden-list",
            lambda: provider.native.model_garden().list_models(
                page_size=1, options=options
            ),
        ),
        ("chat-completions", chat),
    ]
    if args.music_model:

        async def music() -> None:
            result = await provider.media().generate_music(
                model=args.music_model,
                prompt="A brief gentle electronic tone, no vocals.",
                options=RetryOptions(timeout_ms=60000),
            )
            if not result.media or not (
                result.media[0].b64_data or result.media[0].url
            ):
                raise RuntimeError("Missing generated audio")

        reads.append(("lyria-interactions-audio", music))
    if args.partner_model:

        async def partner() -> None:
            result = await provider.native.model_garden().anthropic_messages(
                model=args.partner_model,
                body={
                    "messages": [{"role": "user", "content": "Reply with OK."}],
                    "max_tokens": 64,
                },
                options=options,
            )
            if not result.get("content"):
                raise RuntimeError("Missing partner content")

        reads.append(("anthropic-raw-predict", partner))
    for name, call in reads:
        try:
            await check(name, call)
        except Exception:
            pass
    return int(any(state != "passed" for state in operations.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="global")
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--music-model")
    parser.add_argument("--partner-model")
    parser.add_argument("--cache-only", action="store_true", help="Check only temporary context-cache lifecycle and confirmed cached-token use")
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
