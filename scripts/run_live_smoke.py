from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import create_anthropic, create_gemini, create_ollama, create_openai, create_vertex, generate_text
from zhivex_ai.errors import ZhivexAIError


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _selected_providers() -> set[str] | None:
    raw = os.getenv("ZHIVEX_SMOKE_PROVIDERS")
    if not raw:
        return None
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _want(provider: str, selected: set[str] | None) -> bool:
    return selected is None or provider in selected


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


async def _run_openai() -> tuple[str, bool, str]:
    model = os.getenv("ZHIVEX_SMOKE_OPENAI_MODEL")
    if not os.getenv("OPENAI_API_KEY") or not model:
        return ("openai", False, "skip: set OPENAI_API_KEY and ZHIVEX_SMOKE_OPENAI_MODEL")
    provider = create_openai()
    result = await generate_text(
        model=provider(model),
        prompt="Reply with exactly OPENAI_SMOKE_OK.",
        max_tokens=20,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if result.text.strip() != "OPENAI_SMOKE_OK.":
        raise RuntimeError(f"unexpected response: {result.text!r}")
    return ("openai", True, f"ok: {model}")


async def _run_gemini() -> tuple[str, bool, str]:
    model = os.getenv("ZHIVEX_SMOKE_GEMINI_MODEL")
    if not os.getenv("GEMINI_API_KEY") or not model:
        return ("gemini", False, "skip: set GEMINI_API_KEY and ZHIVEX_SMOKE_GEMINI_MODEL")
    provider = create_gemini()
    result = await generate_text(
        model=provider(model),
        prompt="Reply with exactly GEMINI_SMOKE_OK.",
        max_tokens=20,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if result.text.strip() != "GEMINI_SMOKE_OK.":
        raise RuntimeError(f"unexpected response: {result.text!r}")
    token_count = await provider.tokens().count(model_id=model, prompt="smoke")
    if token_count.total_tokens is None or token_count.total_tokens <= 0:
        raise RuntimeError(f"unexpected token count: {token_count.total_tokens!r}")
    media_details = []
    if _enabled("ZHIVEX_SMOKE_GOOGLE_MEDIA"):
        image_model = os.getenv("ZHIVEX_SMOKE_GEMINI_IMAGE_MODEL")
        video_model = os.getenv("ZHIVEX_SMOKE_GEMINI_VIDEO_MODEL")
        media_model = os.getenv("ZHIVEX_SMOKE_GEMINI_MEDIA_MODEL")
        if image_model:
            image = await provider.images().generate(model=image_model, prompt="A small blue square icon.")
            if not image.images:
                raise RuntimeError("Gemini image smoke returned no images")
            media_details.append(f"image={image_model}")
        if video_model:
            operation = await provider.videos().generate(model=video_model, prompt="A two-second shot of a blue square.")
            if not operation.name:
                raise RuntimeError("Gemini video smoke returned no operation name")
            media_details.append(f"video={video_model}")
        if media_model:
            media = await provider.media().generate_music(model=media_model, prompt="A very short soft synth sting.")
            if not media.media:
                raise RuntimeError("Gemini media smoke returned no media")
            media_details.append(f"media={media_model}")
    suffix = f", {', '.join(media_details)}" if media_details else ""
    return ("gemini", True, f"ok: {model}, tokens={token_count.total_tokens}{suffix}")


async def _run_anthropic() -> tuple[str, bool, str]:
    model = os.getenv("ZHIVEX_SMOKE_ANTHROPIC_MODEL")
    if not os.getenv("ANTHROPIC_API_KEY") or not model:
        return ("anthropic", False, "skip: set ANTHROPIC_API_KEY and ZHIVEX_SMOKE_ANTHROPIC_MODEL")
    provider = create_anthropic()
    result = await generate_text(
        model=provider.native.language_model(model),
        prompt="Reply with exactly ANTHROPIC_SMOKE_OK.",
        max_tokens=20,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if result.text.strip() != "ANTHROPIC_SMOKE_OK.":
        raise RuntimeError(f"unexpected response: {result.text!r}")
    token_count = await provider.tokens().count(model_id=model, prompt="smoke")
    if token_count.total_tokens is None or token_count.total_tokens <= 0:
        raise RuntimeError(f"unexpected token count: {token_count.total_tokens!r}")
    return ("anthropic", True, f"ok: {model}, tokens={token_count.total_tokens}")


async def _run_vertex() -> tuple[str, bool, str]:
    model = os.getenv("ZHIVEX_SMOKE_VERTEX_MODEL")
    access_token = os.getenv("VERTEX_ACCESS_TOKEN") or os.getenv("GOOGLE_ACCESS_TOKEN")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    location = os.getenv("VERTEX_LOCATION", "us-central1")
    if not access_token or not project_id or not model:
        return (
            "vertex",
            False,
            "skip: set VERTEX_ACCESS_TOKEN (or GOOGLE_ACCESS_TOKEN), GOOGLE_CLOUD_PROJECT, and ZHIVEX_SMOKE_VERTEX_MODEL",
        )
    provider = create_vertex(access_token=access_token, project_id=project_id, location=location)
    result = await generate_text(
        model=provider(model),
        prompt="Reply with exactly VERTEX_SMOKE_OK.",
        max_tokens=20,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if result.text.strip() != "VERTEX_SMOKE_OK.":
        raise RuntimeError(f"unexpected response: {result.text!r}")
    token_count = await provider.tokens().count(model_id=model, prompt="smoke")
    if token_count.total_tokens is None or token_count.total_tokens <= 0:
        raise RuntimeError(f"unexpected token count: {token_count.total_tokens!r}")
    media_details = []
    if _enabled("ZHIVEX_SMOKE_GOOGLE_MEDIA"):
        image_model = os.getenv("ZHIVEX_SMOKE_VERTEX_IMAGE_MODEL")
        video_model = os.getenv("ZHIVEX_SMOKE_VERTEX_VIDEO_MODEL")
        media_model = os.getenv("ZHIVEX_SMOKE_VERTEX_MEDIA_MODEL")
        if image_model:
            image = await provider.images().generate(model=image_model, prompt="A small blue square icon.")
            if not image.images:
                raise RuntimeError("Vertex image smoke returned no images")
            media_details.append(f"image={image_model}")
        if video_model:
            operation = await provider.videos().generate(model=video_model, prompt="A two-second shot of a blue square.")
            if not operation.name:
                raise RuntimeError("Vertex video smoke returned no operation name")
            media_details.append(f"video={video_model}")
        if media_model:
            media = await provider.media().generate_music(model=media_model, prompt="A very short soft synth sting.")
            if not media.media:
                raise RuntimeError("Vertex media smoke returned no media")
            media_details.append(f"media={media_model}")
    suffix = f", {', '.join(media_details)}" if media_details else ""
    return ("vertex", True, f"ok: {model}, tokens={token_count.total_tokens}{suffix}")


async def _run_ollama() -> tuple[str, bool, str]:
    model = os.getenv("ZHIVEX_SMOKE_OLLAMA_MODEL")
    base_url = os.getenv("ZHIVEX_SMOKE_OLLAMA_BASE_URL", "http://localhost:11434/v1")
    if not model:
        return ("ollama", False, "skip: set ZHIVEX_SMOKE_OLLAMA_MODEL")
    provider = create_ollama(base_url=base_url)
    result = await generate_text(
        model=provider.native.language_model(model),
        prompt="Reply with exactly OLLAMA_SMOKE_OK.",
        max_tokens=20,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if result.text.strip() != "OLLAMA_SMOKE_OK.":
        raise RuntimeError(f"unexpected response: {result.text!r}")
    return ("ollama", True, f"ok: {model} @ {base_url}")


async def main() -> int:
    _load_dotenv_if_available()
    selected = _selected_providers()
    checks = []
    if _want("openai", selected):
        checks.append(_run_openai)
    if _want("gemini", selected):
        checks.append(_run_gemini)
    if _want("anthropic", selected):
        checks.append(_run_anthropic)
    if _want("vertex", selected):
        checks.append(_run_vertex)
    if _want("ollama", selected):
        checks.append(_run_ollama)

    failures = 0
    for check in checks:
        provider_name = check.__name__.replace("_run_", "")
        try:
            provider, ran, message = await check()
            print(f"[{provider}] {message}")
            if ran:
                continue
        except (ZhivexAIError, RuntimeError, OSError) as error:
            failures += 1
            print(f"[{provider_name}] fail: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
