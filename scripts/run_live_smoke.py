from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
_USE_INSTALLED_PACKAGE = os.getenv("ZHIVEX_SMOKE_USE_INSTALLED", "").strip().lower() in {"1", "true", "yes", "on"}
if not _USE_INSTALLED_PACKAGE and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    Agent,
    AudioInput,
    LanguageModel,
    create_anthropic,
    create_azure_openai,
    create_gemini,
    create_kimi,
    create_ollama,
    create_openai,
    create_qwen,
    create_vertex,
    create_vllm,
    embed,
    generate_speech,
    generate_text,
    run_agent,
    tool,
    transcribe_audio,
)


_PROVIDER_ALIASES = {"azure": "azure-openai"}
_SUPPORTED_PROVIDERS = {
    "openai",
    "gemini",
    "anthropic",
    "azure-openai",
    "vertex",
    "ollama",
    "qwen",
    "kimi",
    "vllm",
}
_URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)


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
    requested = {item.strip().lower() for item in raw.split(",") if item.strip()}
    normalized = {_PROVIDER_ALIASES.get(item, item) for item in requested}
    unknown = sorted(normalized - _SUPPORTED_PROVIDERS)
    if unknown:
        raise ValueError(f"Unknown live smoke provider selector(s): {', '.join(unknown)}")
    if not normalized:
        raise ValueError("ZHIVEX_SMOKE_PROVIDERS did not contain any provider selectors.")
    return normalized


def _want(provider: str, selected: set[str] | None) -> bool:
    return selected is None or provider in selected


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _matches_smoke_token(text: str, expected: str) -> bool:
    return text.strip().rstrip(".") == expected.rstrip(".")


def _safe_error_message(error: BaseException) -> str:
    message = str(error)
    secret_markers = (
        "API_KEY",
        "ACCESS_TOKEN",
        "AUTH_TOKEN",
        "PASSWORD",
        "SECRET",
        "BASE_URL",
        "ENDPOINT",
        "DSN",
        "CREDENTIAL",
    )
    for name, value in os.environ.items():
        if value and any(marker in name.upper() for marker in secret_markers):
            message = message.replace(value, "[REDACTED]")

    def redact_url(match: re.Match[str]) -> str:
        parsed = urlsplit(match.group(0))
        hostname = parsed.hostname or "redacted-host"
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"{parsed.scheme}://{hostname}{port}/[REDACTED]"

    return _URL_RE.sub(redact_url, message)


async def _run_agent_tool_smoke(*, provider: str, model: LanguageModel) -> None:
    nonce = "zhivex-agent-smoke"
    executions: list[dict[str, str]] = []

    def validate_agent_smoke(input: dict[str, str]) -> dict[str, object]:
        executions.append(dict(input))
        return {"nonce": input.get("nonce"), "validated": input.get("nonce") == nonce}

    agent = Agent(
        name=f"{provider.replace('-', '_')}_live_smoke",
        instructions=(
            'Call the "validate_agent_smoke" tool exactly once with '
            f'{{"nonce":"{nonce}"}}. Only after receiving its result, reply with AGENT_SMOKE_OK.'
        ),
        model=model,
        tools={
            "validate_agent_smoke": tool(
                name="validate_agent_smoke",
                description="Validates the fixed, non-secret nonce used by the Zhivex live agent smoke.",
                schema=dict[str, str],
                execute=validate_agent_smoke,
            )
        },
    )
    result = await run_agent(
        agent=agent,
        prompt=(
            'Run the required "validate_agent_smoke" tool now with the exact nonce '
            f'"{nonce}", then reply with AGENT_SMOKE_OK.'
        ),
        max_steps=3,
        temperature=0,
        max_tokens=80,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=30_000,
    )
    matching_results = [item for item in result.tool_results if item.tool_name == "validate_agent_smoke"]
    if executions != [{"nonce": nonce}]:
        raise RuntimeError(f"agent tool executed with unexpected inputs or count: {executions!r}")
    if len(matching_results) != 1 or matching_results[0].is_error:
        raise RuntimeError("agent smoke did not return exactly one successful local tool result")
    if matching_results[0].output != {"nonce": nonce, "validated": True}:
        raise RuntimeError(f"agent smoke returned an unexpected tool result: {matching_results[0].output!r}")
    if not _matches_smoke_token(result.text, "AGENT_SMOKE_OK"):
        raise RuntimeError(f"agent smoke returned unexpected final text: {result.text!r}")


async def _run_openai() -> tuple[str, bool, str, bool]:
    model = os.getenv("ZHIVEX_SMOKE_OPENAI_MODEL")
    if not os.getenv("OPENAI_API_KEY") or not model:
        return ("openai", False, "skip: set OPENAI_API_KEY and ZHIVEX_SMOKE_OPENAI_MODEL", False)
    provider = create_openai()
    language_model = provider(model)
    result = await generate_text(
        model=language_model,
        prompt="Reply with exactly OPENAI_SMOKE_OK.",
        max_tokens=20,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if not _matches_smoke_token(result.text, "OPENAI_SMOKE_OK."):
        raise RuntimeError(f"unexpected response: {result.text!r}")
    agent_ran = _enabled("ZHIVEX_SMOKE_AGENTS")
    if agent_ran:
        await _run_agent_tool_smoke(provider="openai", model=language_model)
    suffix = ", agent-tool=ok" if agent_ran else ""
    return ("openai", True, f"ok: {model}{suffix}", agent_ran)


async def _run_gemini() -> tuple[str, bool, str, bool]:
    model = os.getenv("ZHIVEX_SMOKE_GEMINI_MODEL")
    if not _gemini_api_key() or not model:
        return (
            "gemini",
            False,
            "skip: set GEMINI_API_KEY (or GOOGLE_GENERATIVE_AI_API_KEY or GOOGLE_API_KEY) and ZHIVEX_SMOKE_GEMINI_MODEL",
            False,
        )
    provider = create_gemini()
    language_model = provider(model)
    result = await generate_text(
        model=language_model,
        prompt="Reply with exactly GEMINI_SMOKE_OK.",
        max_tokens=20,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if not _matches_smoke_token(result.text, "GEMINI_SMOKE_OK."):
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
    agent_ran = _enabled("ZHIVEX_SMOKE_AGENTS")
    if agent_ran:
        await _run_agent_tool_smoke(provider="gemini", model=language_model)
        media_details.append("agent-tool=ok")
    suffix = f", {', '.join(media_details)}" if media_details else ""
    return ("gemini", True, f"ok: {model}, tokens={token_count.total_tokens}{suffix}", agent_ran)


async def _run_anthropic() -> tuple[str, bool, str, bool]:
    model = os.getenv("ZHIVEX_SMOKE_ANTHROPIC_MODEL")
    if not os.getenv("ANTHROPIC_API_KEY") or not model:
        return ("anthropic", False, "skip: set ANTHROPIC_API_KEY and ZHIVEX_SMOKE_ANTHROPIC_MODEL", False)
    provider = create_anthropic()
    language_model = provider.native.language_model(model)
    result = await generate_text(
        model=language_model,
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
    agent_ran = _enabled("ZHIVEX_SMOKE_AGENTS")
    if agent_ran:
        await _run_agent_tool_smoke(provider="anthropic", model=language_model)
    suffix = ", agent-tool=ok" if agent_ran else ""
    return ("anthropic", True, f"ok: {model}, tokens={token_count.total_tokens}{suffix}", agent_ran)


async def _run_azure_openai() -> tuple[str, bool, str, bool]:
    model = os.getenv("ZHIVEX_SMOKE_AZURE_OPENAI_MODEL")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key or not endpoint or not model:
        return (
            "azure-openai",
            False,
            "skip: set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and ZHIVEX_SMOKE_AZURE_OPENAI_MODEL",
            False,
        )
    provider = create_azure_openai(api_key=api_key, endpoint=endpoint)
    language_model = provider(model)
    result = await generate_text(
        model=language_model,
        prompt="Reply with exactly AZURE_OPENAI_SMOKE_OK.",
        max_tokens=20,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if result.text.strip() != "AZURE_OPENAI_SMOKE_OK.":
        raise RuntimeError(f"unexpected response: {result.text!r}")
    agent_ran = _enabled("ZHIVEX_SMOKE_AGENTS")
    if agent_ran:
        await _run_agent_tool_smoke(provider="azure-openai", model=language_model)
    suffix = ", agent-tool=ok" if agent_ran else ""
    return ("azure-openai", True, f"ok: {model}{suffix}", agent_ran)


async def _run_vertex() -> tuple[str, bool, str, bool]:
    model = os.getenv("ZHIVEX_SMOKE_VERTEX_MODEL")
    access_token = os.getenv("VERTEX_ACCESS_TOKEN") or os.getenv("GOOGLE_ACCESS_TOKEN")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    location = os.getenv("VERTEX_LOCATION", "us-central1")
    if not access_token or not project_id or not model:
        return (
            "vertex",
            False,
            "skip: set VERTEX_ACCESS_TOKEN (or GOOGLE_ACCESS_TOKEN), GOOGLE_CLOUD_PROJECT, and ZHIVEX_SMOKE_VERTEX_MODEL",
            False,
        )
    provider = create_vertex(access_token=access_token, project_id=project_id, location=location)
    language_model = provider(model)
    result = await generate_text(
        model=language_model,
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
    agent_ran = _enabled("ZHIVEX_SMOKE_AGENTS")
    if agent_ran:
        await _run_agent_tool_smoke(provider="vertex", model=language_model)
        media_details.append("agent-tool=ok")
    suffix = f", {', '.join(media_details)}" if media_details else ""
    return ("vertex", True, f"ok: {model}, tokens={token_count.total_tokens}{suffix}", agent_ran)


async def _run_ollama() -> tuple[str, bool, str, bool]:
    model = os.getenv("ZHIVEX_SMOKE_OLLAMA_MODEL")
    base_url = os.getenv("ZHIVEX_SMOKE_OLLAMA_BASE_URL", "http://localhost:11434/v1")
    if not model:
        return ("ollama", False, "skip: set ZHIVEX_SMOKE_OLLAMA_MODEL", False)
    provider = create_ollama(base_url=base_url)
    language_model = provider.native.language_model(model)
    result = await generate_text(
        model=language_model,
        prompt="Reply with exactly OLLAMA_SMOKE_OK.",
        max_tokens=20,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if result.text.strip() != "OLLAMA_SMOKE_OK.":
        raise RuntimeError(f"unexpected response: {result.text!r}")
    agent_ran = _enabled("ZHIVEX_SMOKE_AGENTS")
    if agent_ran:
        await _run_agent_tool_smoke(provider="ollama", model=language_model)
    suffix = ", agent-tool=ok" if agent_ran else ""
    return ("ollama", True, f"ok: {model}{suffix}", agent_ran)


async def _run_qwen() -> tuple[str, bool, str, bool]:
    model = os.getenv("ZHIVEX_SMOKE_QWEN_MODEL")
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    region = os.getenv("ZHIVEX_SMOKE_QWEN_REGION", "intl")
    base_url = os.getenv("ZHIVEX_SMOKE_QWEN_BASE_URL")
    responses_base_url = os.getenv("ZHIVEX_SMOKE_QWEN_RESPONSES_BASE_URL")
    if not api_key or not model:
        return (
            "qwen",
            False,
            "skip: set DASHSCOPE_API_KEY (or QWEN_API_KEY) and ZHIVEX_SMOKE_QWEN_MODEL",
            False,
        )
    provider = create_qwen(
        api_key=api_key,
        region=region,  # type: ignore[arg-type]
        base_url=base_url,
        responses_base_url=responses_base_url,
    )
    language_model = provider(model)
    result = await generate_text(
        model=language_model,
        prompt="Reply with exactly QWEN_SMOKE_OK.",
        max_tokens=20,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if result.text.strip() != "QWEN_SMOKE_OK.":
        raise RuntimeError(f"unexpected response: {result.text!r}")

    details = [f"ok: {model}", f"region={region}"]
    embedding_model = os.getenv("ZHIVEX_SMOKE_QWEN_EMBEDDING_MODEL")
    if embedding_model:
        embedding = await embed(
            model=provider.native.embedding_model(embedding_model),
            value="smoke",
            max_retries=1,
            retry_backoff_ms=250,
            timeout_ms=20_000,
        )
        if not embedding.embedding:
            raise RuntimeError("Qwen embedding smoke returned an empty vector")
        details.append(f"embedding={embedding_model}")

    asr_model = os.getenv("ZHIVEX_SMOKE_QWEN_ASR_MODEL")
    asr_audio_path = os.getenv("ZHIVEX_SMOKE_QWEN_ASR_AUDIO_PATH")
    if asr_model and asr_audio_path:
        audio_path = Path(asr_audio_path)
        if not audio_path.exists():
            raise RuntimeError(f"Qwen ASR audio path does not exist: {asr_audio_path}")
        transcript = await transcribe_audio(
            model=provider.native.transcription_model(asr_model),
            audio=AudioInput(data=audio_path.read_bytes(), media_type=os.getenv("ZHIVEX_SMOKE_QWEN_ASR_MEDIA_TYPE", "audio/wav"), filename=audio_path.name),
            max_retries=1,
            retry_backoff_ms=250,
            timeout_ms=30_000,
        )
        if not transcript.text.strip():
            raise RuntimeError("Qwen ASR smoke returned empty text")
        details.append(f"asr={asr_model}")
    elif asr_model:
        details.append(f"asr={asr_model}:skip-audio")

    tts_model = os.getenv("ZHIVEX_SMOKE_QWEN_TTS_MODEL")
    if tts_model:
        speech = await generate_speech(
            model=provider.native.speech_model(tts_model),
            input="Zhivex Qwen smoke test.",
            provider_options={"language_type": os.getenv("ZHIVEX_SMOKE_QWEN_TTS_LANGUAGE_TYPE", "English")},
            max_retries=1,
            retry_backoff_ms=250,
            timeout_ms=30_000,
        )
        if not speech.audio:
            raise RuntimeError("Qwen TTS smoke returned empty audio")
        details.append(f"tts={tts_model}")

    agent_ran = _enabled("ZHIVEX_SMOKE_AGENTS")
    if agent_ran:
        await _run_agent_tool_smoke(provider="qwen", model=language_model)
        details.append("agent-tool=ok")
    return ("qwen", True, ", ".join(details), agent_ran)


async def _run_kimi() -> tuple[str, bool, str, bool]:
    model = os.getenv("ZHIVEX_SMOKE_KIMI_MODEL")
    api_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
    base_url = os.getenv("ZHIVEX_SMOKE_KIMI_BASE_URL") or os.getenv("MOONSHOT_BASE_URL")
    if not api_key or not model:
        return (
            "kimi",
            False,
            "skip: set MOONSHOT_API_KEY (or KIMI_API_KEY) and ZHIVEX_SMOKE_KIMI_MODEL",
            False,
        )
    provider = create_kimi(api_key=api_key, base_url=base_url)
    language_model = provider(model)
    result = await generate_text(
        model=language_model,
        prompt="Reply with exactly KIMI_SMOKE_OK.",
        max_tokens=20,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if result.text.strip() != "KIMI_SMOKE_OK.":
        raise RuntimeError(f"unexpected response: {result.text!r}")
    agent_ran = _enabled("ZHIVEX_SMOKE_AGENTS")
    if agent_ran:
        await _run_agent_tool_smoke(provider="kimi", model=language_model)
    suffix = ", agent-tool=ok" if agent_ran else ""
    return ("kimi", True, f"ok: {model}{suffix}", agent_ran)


async def _run_vllm() -> tuple[str, bool, str, bool]:
    model = os.getenv("ZHIVEX_SMOKE_VLLM_MODEL")
    base_url = os.getenv("ZHIVEX_SMOKE_VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("ZHIVEX_SMOKE_VLLM_API_KEY") or os.getenv("VLLM_API_KEY")
    if not model:
        return ("vllm", False, "skip: set ZHIVEX_SMOKE_VLLM_MODEL", False)
    provider = create_vllm(api_key=api_key, base_url=base_url)
    language_model = provider(model)
    result = await generate_text(
        model=language_model,
        prompt="Reply with exactly VLLM_SMOKE_OK.",
        max_tokens=20,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if result.text.strip() != "VLLM_SMOKE_OK.":
        raise RuntimeError(f"unexpected response: {result.text!r}")
    agent_ran = _enabled("ZHIVEX_SMOKE_AGENTS")
    if agent_ran:
        await _run_agent_tool_smoke(provider="vllm", model=language_model)
    suffix = ", agent-tool=ok" if agent_ran else ""
    return ("vllm", True, f"ok: {model}{suffix}", agent_ran)


async def main() -> int:
    _load_dotenv_if_available()
    try:
        selected = _selected_providers()
    except ValueError as error:
        print(f"[smoke] fail: {error}")
        return 2
    strict = _enabled("ZHIVEX_SMOKE_STRICT")
    agent_smoke_enabled = _enabled("ZHIVEX_SMOKE_AGENTS")
    checks = []
    if _want("openai", selected):
        checks.append(_run_openai)
    if _want("gemini", selected):
        checks.append(_run_gemini)
    if _want("anthropic", selected):
        checks.append(_run_anthropic)
    if _want("azure-openai", selected):
        checks.append(_run_azure_openai)
    if _want("vertex", selected):
        checks.append(_run_vertex)
    if _want("ollama", selected):
        checks.append(_run_ollama)
    if _want("qwen", selected):
        checks.append(_run_qwen)
    if _want("kimi", selected):
        checks.append(_run_kimi)
    if _want("vllm", selected):
        checks.append(_run_vllm)

    failures = 0
    executed = 0
    agent_executed = 0
    for check in checks:
        provider_name = check.__name__.replace("_run_", "")
        try:
            provider, ran, message, agent_ran = await check()
            print(f"[{provider}] {message}")
            if ran:
                executed += 1
                if agent_ran:
                    agent_executed += 1
                continue
        except Exception as error:
            failures += 1
            print(f"[{provider_name}] fail: {_safe_error_message(error)}")
    if strict and executed == 0:
        failures += 1
        print("[smoke] fail: strict mode requires at least one provider smoke to execute.")
    elif strict and agent_smoke_enabled and agent_executed == 0:
        failures += 1
        print("[smoke] fail: strict agent mode requires at least one agent tool smoke to execute.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
