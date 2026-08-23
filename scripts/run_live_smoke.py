from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
_USE_INSTALLED_PACKAGE = os.getenv("ZHIVEX_SMOKE_USE_INSTALLED", "").strip().lower() in {"1", "true", "yes", "on"}
if not _USE_INSTALLED_PACKAGE and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    Agent,
    AudioInput,
    LanguageModel,
    PortableDocument,
    PortableRetrievalConfig,
    ReasoningConfig,
    create_anthropic,
    create_azure_openai,
    create_deepseek,
    create_gemini,
    create_kimi,
    create_meta,
    create_ollama,
    create_openai,
    create_qwen,
    create_vertex,
    create_vllm,
    embed,
    generate_object,
    generate_speech,
    generate_text,
    run_agent,
    tool,
    transcribe_audio,
    stream_text,
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
    "deepseek",
    "meta",
    "vllm",
}
_URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_PROVIDER_MODEL_ENV = {
    "openai": "ZHIVEX_SMOKE_OPENAI_MODEL",
    "gemini": "ZHIVEX_SMOKE_GEMINI_MODEL",
    "anthropic": "ZHIVEX_SMOKE_ANTHROPIC_MODEL",
    "azure-openai": "ZHIVEX_SMOKE_AZURE_OPENAI_MODEL",
    "vertex": "ZHIVEX_SMOKE_VERTEX_MODEL",
    "ollama": "ZHIVEX_SMOKE_OLLAMA_MODEL",
    "qwen": "ZHIVEX_SMOKE_QWEN_MODEL",
    "kimi": "ZHIVEX_SMOKE_KIMI_MODEL",
    "deepseek": "ZHIVEX_SMOKE_DEEPSEEK_MODEL",
    "meta": "ZHIVEX_SMOKE_META_MODEL",
    "vllm": "ZHIVEX_SMOKE_VLLM_MODEL",
}
_RELEASE_SMOKE_OPERATIONS = frozenset(
    {"generation", "streaming", "structured-output", "portable-retrieval", "agent-tool"}
)


class _AgentSmokeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nonce: str


class _MetaStructuredSmokeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nonce: str


def _source_package_version() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    return str(pyproject["project"]["version"])


def _observed_package_version() -> str:
    if _enabled("ZHIVEX_SMOKE_USE_INSTALLED"):
        return metadata.version("zhivex-ai-sdk")
    return _source_package_version()


def _release_artifact_details() -> dict[str, str] | None:
    raw_path = os.getenv("ZHIVEX_SMOKE_ARTIFACT_PATH")
    if not raw_path:
        return None
    artifact_path = Path(raw_path)
    if artifact_path.is_dir():
        candidates = sorted(artifact_path.glob("zhivex_ai_sdk-*.whl"))
        if len(candidates) != 1:
            raise ValueError(
                "ZHIVEX_SMOKE_ARTIFACT_PATH must contain exactly one zhivex_ai_sdk wheel."
            )
        artifact_path = candidates[0]
    if not artifact_path.is_file():
        raise ValueError("ZHIVEX_SMOKE_ARTIFACT_PATH must point to a release wheel or its directory.")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return {"filename": artifact_path.name, "sha256": digest}


def _load_release_smoke_policy() -> dict[str, Any] | None:
    raw_path = os.getenv("ZHIVEX_RELEASE_SMOKE_POLICY")
    if not raw_path:
        return None
    path = Path(raw_path)
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load release smoke policy {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Release smoke policy must be a schema_version 1 JSON object.")
    if not isinstance(payload.get("package_version"), str):
        raise ValueError("Release smoke policy package_version must be a string.")
    providers = payload.get("required_providers")
    if not isinstance(providers, dict) or not providers:
        raise ValueError("Release smoke policy required_providers must be a non-empty object.")
    for provider, requirement in providers.items():
        if provider not in _SUPPORTED_PROVIDERS or not isinstance(requirement, dict):
            raise ValueError(f'Invalid release smoke provider requirement: "{provider}".')
        operations = requirement.get("operations")
        if not isinstance(operations, list) or not operations:
            raise ValueError(f'Release smoke provider "{provider}" must require operations.')
        unknown_operations = set(operations) - _RELEASE_SMOKE_OPERATIONS
        if unknown_operations:
            raise ValueError(
                f'Release smoke provider "{provider}" has unknown operations: '
                f"{', '.join(sorted(unknown_operations))}."
            )
        model = requirement.get("model")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError(f'Release smoke provider "{provider}" model must be a string.')
    return payload


def _validate_release_smoke_configuration(
    policy: dict[str, Any],
    *,
    selected: set[str] | None,
) -> None:
    expected_version = str(policy["package_version"])
    observed_version = _observed_package_version()
    if observed_version != expected_version:
        raise ValueError(
            f"Release smoke package version mismatch: expected {expected_version}, observed {observed_version}."
        )
    required_providers = set(policy["required_providers"])
    if selected is None or not required_providers.issubset(selected):
        missing = required_providers - (selected or set())
        raise ValueError(
            "Release smoke policy requires explicit provider selection; "
            f"missing: {', '.join(sorted(missing or required_providers))}."
        )
    if not _enabled("ZHIVEX_SMOKE_STRICT"):
        raise ValueError("Release smoke policy requires ZHIVEX_SMOKE_STRICT=1.")
    required_operations = {
        str(operation)
        for requirement in policy["required_providers"].values()
        for operation in requirement["operations"]
    }
    if "agent-tool" in required_operations and not _enabled("ZHIVEX_SMOKE_AGENTS"):
        raise ValueError("Release smoke policy requires ZHIVEX_SMOKE_AGENTS=1.")
    meta_operations = set(
        policy["required_providers"].get("meta", {}).get("operations", [])
    )
    if meta_operations - {"generation", "agent-tool"} and not _enabled(
        "ZHIVEX_SMOKE_META_CERTIFICATION"
    ):
        raise ValueError("Release smoke policy requires ZHIVEX_SMOKE_META_CERTIFICATION=1.")
    for provider, requirement in policy["required_providers"].items():
        expected_model = requirement.get("model")
        if expected_model is None:
            continue
        observed_model = os.getenv(_PROVIDER_MODEL_ENV[provider])
        if observed_model != expected_model:
            raise ValueError(
                f'Release smoke provider "{provider}" requires model "{expected_model}", '
                f'observed "{observed_model or "unset"}".'
            )
    if policy.get("require_installed_package") and not _enabled("ZHIVEX_SMOKE_USE_INSTALLED"):
        raise ValueError("Release smoke policy requires ZHIVEX_SMOKE_USE_INSTALLED=1.")
    artifact = _release_artifact_details()
    if policy.get("require_artifact_sha256") and artifact is None:
        raise ValueError("Release smoke policy requires ZHIVEX_SMOKE_ARTIFACT_PATH.")


def _operations_for_outcome(provider: str, *, agent_ran: bool) -> set[str]:
    operations = {"generation"}
    if agent_ran:
        operations.add("agent-tool")
    if provider == "meta" and _enabled("ZHIVEX_SMOKE_META_CERTIFICATION"):
        operations.update({"streaming", "structured-output", "portable-retrieval"})
    return operations


def _source_revision() -> str:
    github_sha = os.getenv("GITHUB_SHA")
    if github_sha:
        return github_sha
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else "unavailable"


def _write_release_smoke_evidence(
    *,
    policy: dict[str, Any] | None,
    executed_operations: dict[str, set[str]],
    failures: int,
) -> None:
    raw_path = os.getenv("ZHIVEX_SMOKE_EVIDENCE_PATH")
    if not raw_path:
        return
    evidence_path = Path(raw_path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = _release_artifact_details()
    payload = {
        "schema_version": 1,
        "status": "passed" if failures == 0 else "failed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "package_version": _observed_package_version(),
        "source_revision": _source_revision(),
        "artifact": artifact,
        "policy_package_version": policy.get("package_version") if policy else None,
        "providers": {
            provider: {
                "model": os.getenv(_PROVIDER_MODEL_ENV[provider]),
                "operations": sorted(operations),
            }
            for provider, operations in sorted(executed_operations.items())
        },
    }
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")


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


def _agent_smoke_generation_options(provider: str) -> dict[str, int | None]:
    if provider == "anthropic":
        return {"temperature": None, "max_tokens": 4096}
    if provider == "meta":
        return {"temperature": None, "max_tokens": 1024}
    if provider == "openai":
        return {"temperature": None, "max_tokens": 512}
    if provider == "gemini":
        return {"temperature": None, "max_tokens": 512}
    return {"temperature": 0, "max_tokens": 80}


def _openai_smoke_reasoning(model_id: str) -> ReasoningConfig | None:
    if model_id.startswith("gpt-5.6"):
        return ReasoningConfig(effort="none")
    return None


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

    def validate_agent_smoke(input: _AgentSmokeInput) -> dict[str, object]:
        executions.append(input.model_dump())
        return {"nonce": input.nonce, "validated": input.nonce == nonce}

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
                schema=_AgentSmokeInput,
                execute=validate_agent_smoke,
            )
        },
    )
    generation_options = _agent_smoke_generation_options(provider)
    result = await run_agent(
        agent=agent,
        prompt=(
            'Run the required "validate_agent_smoke" tool now with the exact nonce '
            f'"{nonce}", then reply with AGENT_SMOKE_OK.'
        ),
        max_steps=3,
        temperature=generation_options["temperature"],
        max_tokens=generation_options["max_tokens"],
        reasoning=(
            ReasoningConfig(effort="none")
            if provider == "qwen"
            else _openai_smoke_reasoning(model.model_id)
            if provider == "openai"
            else ReasoningConfig(effort="low")
            if provider == "meta"
            else None
        ),
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
        reasoning=_openai_smoke_reasoning(model),
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
        max_tokens=256,
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
        max_tokens=1024,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if not _matches_smoke_token(result.text, "ANTHROPIC_SMOKE_OK."):
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
        reasoning=ReasoningConfig(effort="none"),
        max_tokens=20,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if not _matches_smoke_token(result.text, "QWEN_SMOKE_OK."):
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


async def _run_deepseek() -> tuple[str, bool, str, bool]:
    model = os.getenv("ZHIVEX_SMOKE_DEEPSEEK_MODEL")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("ZHIVEX_SMOKE_DEEPSEEK_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
    if not api_key or not model:
        return (
            "deepseek",
            False,
            "skip: set DEEPSEEK_API_KEY and ZHIVEX_SMOKE_DEEPSEEK_MODEL",
            False,
        )
    provider = create_deepseek(api_key=api_key, base_url=base_url)
    language_model = provider(model)
    result = await generate_text(
        model=language_model,
        prompt="Reply with exactly DEEPSEEK_SMOKE_OK.",
        reasoning=ReasoningConfig(effort="none"),
        max_tokens=20,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if not _matches_smoke_token(result.text, "DEEPSEEK_SMOKE_OK"):
        raise RuntimeError(f"unexpected response: {result.text!r}")
    agent_ran = _enabled("ZHIVEX_SMOKE_AGENTS")
    if agent_ran:
        await _run_agent_tool_smoke(provider="deepseek", model=language_model)
    suffix = ", agent-tool=ok" if agent_ran else ""
    return ("deepseek", True, f"ok: {model}{suffix}", agent_ran)


async def _run_meta_portable_certification(*, model: LanguageModel) -> None:
    streamed = await stream_text(
        model=model,
        prompt="Reply with exactly META_STREAM_SMOKE_OK.",
        reasoning=ReasoningConfig(effort="low"),
        max_tokens=512,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    ).collect()
    if not _matches_smoke_token(streamed.text, "META_STREAM_SMOKE_OK"):
        raise RuntimeError(f"Meta streaming smoke returned unexpected text: {streamed.text!r}")

    structured = await generate_object(
        model=model,
        prompt='Return only JSON with the exact shape {"nonce":"meta-structured-smoke"}.',
        schema=_MetaStructuredSmokeOutput,
        schema_name="meta_release_smoke",
        reasoning=ReasoningConfig(effort="low"),
        max_tokens=512,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if structured.object.nonce != "meta-structured-smoke":
        raise RuntimeError(
            f"Meta structured-output smoke returned unexpected nonce: {structured.object.nonce!r}"
        )

    retrieval = await generate_text(
        model=model,
        prompt="Using only the supplied release document, reply with exactly META_RETRIEVAL_SMOKE_OK.",
        retrieval=PortableRetrievalConfig(
            documents=[
                PortableDocument(
                    document_id="meta-release-smoke",
                    title="Release certification marker",
                    text="The exact required reply is META_RETRIEVAL_SMOKE_OK.",
                )
            ],
            max_documents=1,
            max_document_chars=256,
        ),
        reasoning=ReasoningConfig(effort="low"),
        max_tokens=512,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if not _matches_smoke_token(retrieval.text, "META_RETRIEVAL_SMOKE_OK"):
        raise RuntimeError(f"Meta retrieval smoke returned unexpected text: {retrieval.text!r}")


async def _run_meta() -> tuple[str, bool, str, bool]:
    model = os.getenv("ZHIVEX_SMOKE_META_MODEL")
    api_key = os.getenv("MODEL_API_KEY")
    base_url = os.getenv("ZHIVEX_SMOKE_META_BASE_URL") or os.getenv("META_BASE_URL")
    if not api_key or not model:
        return (
            "meta",
            False,
            "skip: set MODEL_API_KEY and ZHIVEX_SMOKE_META_MODEL",
            False,
        )
    provider = create_meta(api_key=api_key, **({"base_url": base_url} if base_url else {}))
    language_model = provider(model)
    result = await generate_text(
        model=language_model,
        prompt="Reply with exactly META_SMOKE_OK.",
        reasoning=ReasoningConfig(effort="low"),
        max_tokens=512,
        max_retries=1,
        retry_backoff_ms=250,
        timeout_ms=20_000,
    )
    if not _matches_smoke_token(result.text, "META_SMOKE_OK"):
        raise RuntimeError(f"unexpected response: {result.text!r}")
    certification_ran = _enabled("ZHIVEX_SMOKE_META_CERTIFICATION")
    if certification_ran:
        await _run_meta_portable_certification(model=language_model)
    agent_ran = _enabled("ZHIVEX_SMOKE_AGENTS")
    if agent_ran:
        await _run_agent_tool_smoke(provider="meta", model=language_model)
    details = []
    if certification_ran:
        details.append("portable-certification=ok")
    if agent_ran:
        details.append("agent-tool=ok")
    suffix = f", {', '.join(details)}" if details else ""
    return ("meta", True, f"ok: {model}{suffix}", agent_ran)


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
        policy = _load_release_smoke_policy()
        if policy is not None:
            _validate_release_smoke_configuration(policy, selected=selected)
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
    if _want("deepseek", selected):
        checks.append(_run_deepseek)
    if _want("meta", selected):
        checks.append(_run_meta)
    if _want("vllm", selected):
        checks.append(_run_vllm)

    failures = 0
    executed_providers: set[str] = set()
    agent_executed_providers: set[str] = set()
    executed_operations: dict[str, set[str]] = {}
    for check in checks:
        provider_name = check.__name__.replace("_run_", "")
        try:
            provider, ran, message, agent_ran = await check()
            print(f"[{provider}] {message}")
            if ran:
                executed_providers.add(provider)
                executed_operations[provider] = _operations_for_outcome(
                    provider,
                    agent_ran=agent_ran,
                )
                if agent_ran:
                    agent_executed_providers.add(provider)
                continue
        except Exception as error:
            failures += 1
            print(f"[{provider_name}] fail: {_safe_error_message(error)}")
    if strict and selected is not None:
        missing_provider_smokes = selected - executed_providers
        if missing_provider_smokes:
            failures += 1
            print(
                "[smoke] fail: strict mode requires every explicitly selected provider smoke to execute; "
                f"missing: {', '.join(sorted(missing_provider_smokes))}."
            )
        if agent_smoke_enabled:
            missing_agent_smokes = executed_providers - agent_executed_providers
            if missing_agent_smokes:
                failures += 1
                print(
                    "[smoke] fail: strict agent mode requires an agent tool smoke for every selected provider "
                    f"that executed; missing: {', '.join(sorted(missing_agent_smokes))}."
                )
    elif strict and not executed_providers:
        failures += 1
        print("[smoke] fail: strict mode requires at least one provider smoke to execute.")
    elif strict and agent_smoke_enabled and not agent_executed_providers:
        failures += 1
        print("[smoke] fail: strict agent mode requires at least one agent tool smoke to execute.")
    if policy is not None:
        for provider, requirement in policy["required_providers"].items():
            missing_operations = set(requirement["operations"]) - executed_operations.get(
                provider,
                set(),
            )
            if missing_operations:
                failures += 1
                print(
                    f'[smoke] fail: release policy operations missing for "{provider}": '
                    f"{', '.join(sorted(missing_operations))}."
                )
    try:
        _write_release_smoke_evidence(
            policy=policy,
            executed_operations=executed_operations,
            failures=failures,
        )
    except (OSError, ValueError) as error:
        failures += 1
        print(f"[smoke] fail: could not write release evidence: {_safe_error_message(error)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
