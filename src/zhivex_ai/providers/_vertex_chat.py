"""Vertex OpenAI-compatible inference, separate from Gemini generateContent."""

from __future__ import annotations

from collections.abc import AsyncIterable
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from typing import Any

from .._http import Fetcher
from .._sse import parse_sse
from ..errors import ProviderHTTPError, UnsupportedFeatureError, ValidationError
from ..messages import normalize_finish_reason, validate_message_parts
from ..runtime import with_retry
from ..schema import create_schema_adapter
from ..types import (
    AgentCapabilities,
    GenerateResult,
    ImagePart,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    ProviderDataPart,
    StreamEvent,
    StreamFinishEvent,
    StreamProviderDataEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    TextPart,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    ToolResultPart,
)
from ._payload import drop_none

GEMMA_MAAS_MODEL = "google/gemma-4-26b-a4b-it-maas"
MISTRAL_VERTEX_MODELS = frozenset({
    "mistralai/mistral-medium-3", "mistralai/mistral-small-2503", "mistralai/codestral-2",
})
VERTEX_MAAS_TEXT_MODELS = frozenset({
    "openai/gpt-oss-120b-maas",
    "openai/gpt-oss-20b-maas",
    "deepseek-ai/deepseek-v3.2-maas",
    "deepseek-ai/deepseek-v3.1-maas",
    "deepseek-ai/deepseek-v4.1-flash-maas",
    "deepseek-ai/deepseek-r1-0528-maas",
    "deepseek-ai/deepseek-ocr-maas",
    "moonshotai/kimi-k2-thinking-maas",
    "zai-org/glm-5-maas",
    "zai-org/glm-5.2-maas",
    "zai-org/glm-4.7-maas",
    "qwen/qwen3-next-80b-a3b-instruct-maas",
    "qwen/qwen3-next-80b-a3b-thinking-maas",
    "qwen/qwen3-235b-a22b-instruct-2507-maas",
    "qwen/qwen3-coder-480b-a35b-instruct-maas",
    "minimaxai/minimax-m2-maas",
    "meta/llama-3.3-70b-instruct-maas",
    "meta/llama-4-maverick-17b-128e-instruct-maas",
    "meta/llama-4-scout-17b-16e-instruct-maas",
    "xai/grok-4.6",
    "xai/grok-4.3",
    "xai/grok-4.20-reasoning",
    "xai/grok-4.20-non-reasoning",
    "xai/grok-4.1-fast-reasoning",
    "xai/grok-4.1-fast-non-reasoning",
})
VERTEX_E5_EMBEDDING_MODELS = frozenset({
    "intfloat/multilingual-e5-small-maas",
    "intfloat/multilingual-e5-large-instruct-maas",
})
VERTEX_MAAS_TEXT_CAPABILITIES = ModelCapabilities(
    streaming=True, tools=False, structured_output=False, json_mode=False,
    tool_choice=False, parallel_tool_calls=False, vision=False, files=False,
    audio_input=False, audio_output=False, embeddings=False, reasoning=False,
    web_search=False,
)


def resolve_maas_model(model_id: str) -> str:
    for model in VERTEX_MAAS_TEXT_MODELS | MISTRAL_VERTEX_MODELS | {GEMMA_MAAS_MODEL}:
        if model_id in {model, model.split("/", 1)[1]}:
            return model
    return model_id


GEMMA_MAAS_CAPABILITIES = ModelCapabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    vision=True,
    parallel_tool_calls=False,
    files=False,
    audio_input=False,
    audio_output=False,
    embeddings=False,
    reasoning=False,
    web_search=False,
    agent_capabilities=AgentCapabilities(support_tier="tier-b", tool_choice_none=True),
)

GLM_52_MAAS_MODEL = "zai-org/glm-5.2-maas"
GLM_52_MAAS_CAPABILITIES = replace(GEMMA_MAAS_CAPABILITIES, vision=False)


@dataclass(frozen=True, slots=True)
class _PartnerChatProfile:
    """Open-model serving contract from the Vertex thinking guide.

    ``thinking`` is ``effort`` (reasoning_effort), ``thinking`` (DeepSeek hybrid),
    or ``enable_thinking`` (chat template). Empty means thinking is model-defined.
    https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/capabilities/thinking
    """

    tools: bool = True
    structured_output: bool = True
    json_mode: bool = True
    vision: bool = False
    reasoning: bool = False
    thinking: str | None = None


def _partner_capabilities(profile: _PartnerChatProfile) -> ModelCapabilities:
    return ModelCapabilities(
        streaming=True,
        tools=profile.tools,
        structured_output=profile.structured_output,
        json_mode=profile.json_mode,
        tool_choice=profile.tools,
        parallel_tool_calls=profile.tools,
        vision=profile.vision,
        files=False,
        audio_input=False,
        audio_output=False,
        embeddings=False,
        reasoning=profile.reasoning,
        web_search=False,
        agent_capabilities=AgentCapabilities(support_tier="tier-b", tool_choice_none=profile.tools),
    )


def _profile(
    *,
    vision: bool = False,
    reasoning: bool = False,
    thinking: str | None = None,
    tools: bool = True,
) -> _PartnerChatProfile:
    return _PartnerChatProfile(
        tools=tools,
        structured_output=tools,
        json_mode=tools,
        vision=vision,
        reasoning=reasoning,
        thinking=thinking,
    )


# Profiles for models added after the original MaaS set. Existing ids keep
# maas_capabilities() below so their published flags do not move.
VERTEX_PARTNER_CHAT_PROFILES: dict[str, _PartnerChatProfile] = {
    "meta/llama-3.3-70b-instruct-maas": _profile(),
    "qwen/qwen3-235b-a22b-instruct-2507-maas": _profile(),
    "qwen/qwen3-coder-480b-a35b-instruct-maas": _profile(),
    "meta/llama-4-maverick-17b-128e-instruct-maas": _profile(vision=True),
    "meta/llama-4-scout-17b-16e-instruct-maas": _profile(vision=True),
    "xai/grok-4.20-non-reasoning": _profile(vision=True),
    "xai/grok-4.1-fast-non-reasoning": _profile(vision=True),
    "deepseek-ai/deepseek-r1-0528-maas": _profile(reasoning=True),
    "qwen/qwen3-next-80b-a3b-thinking-maas": _profile(reasoning=True),
    "openai/gpt-oss-20b-maas": _profile(reasoning=True, thinking="effort"),
    "deepseek-ai/deepseek-v3.1-maas": _profile(reasoning=True, thinking="thinking"),
    "zai-org/glm-4.7-maas": _profile(reasoning=True, thinking="enable_thinking"),
    "xai/grok-4.6": _profile(vision=True, reasoning=True),
    "xai/grok-4.3": _profile(vision=True, reasoning=True),
    "xai/grok-4.20-reasoning": _profile(vision=True, reasoning=True),
    "xai/grok-4.1-fast-reasoning": _profile(vision=True, reasoning=True),
    "deepseek-ai/deepseek-ocr-maas": _PartnerChatProfile(
        tools=False, structured_output=False, json_mode=False, vision=True,
    ),
    "deepseek-ai/deepseek-v4.1-flash-maas": _PartnerChatProfile(
        tools=False, structured_output=False, json_mode=False,
    ),
}


def maas_capabilities(model_id: str) -> ModelCapabilities:
    profile = VERTEX_PARTNER_CHAT_PROFILES.get(model_id)
    if profile is not None:
        return _partner_capabilities(profile)
    if model_id == "mistralai/mistral-medium-3":
        return replace(GEMMA_MAAS_CAPABILITIES, structured_output=False, json_mode=False)
    if model_id == "mistralai/mistral-small-2503":
        return replace(VERTEX_MAAS_TEXT_CAPABILITIES, vision=True)
    if model_id == GEMMA_MAAS_MODEL:
        return GEMMA_MAAS_CAPABILITIES
    if model_id == GLM_52_MAAS_MODEL:
        return GLM_52_MAAS_CAPABILITIES
    if model_id == "openai/gpt-oss-120b-maas":
        return replace(GEMMA_MAAS_CAPABILITIES, vision=False)
    return VERTEX_MAAS_TEXT_CAPABILITIES


def _messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        content = []
        calls = []
        reasoning = []
        for part in message.parts:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                content.append({"type": "image_url", "image_url": {"url": part.image}})
            elif isinstance(part, ToolCallPart):
                calls.append(
                    {
                        "id": part.tool_call.id,
                        "type": "function",
                        "function": {
                            "name": part.tool_call.name,
                            "arguments": json.dumps(part.tool_call.input),
                        },
                    }
                )
            elif isinstance(part, ToolResultPart):
                tool_result = part.tool_result
                output = (
                    {"error": tool_result.error.message}
                    if tool_result.is_error and tool_result.error
                    else tool_result.output
                )
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result.tool_call_id,
                        "content": json.dumps(output),
                    }
                )
            elif isinstance(part, ProviderDataPart) and part.provider == "vertex":
                if isinstance(part.data, dict) and isinstance(
                    part.data.get("reasoning_content"), str
                ):
                    reasoning.append(part.data["reasoning_content"])
        if content or calls or reasoning:
            result.append(
                drop_none(
                    {
                        "role": message.role,
                        "content": "".join(item["text"] for item in content)
                        if all(item["type"] == "text" for item in content)
                        else content,
                        "tool_calls": calls or None,
                        "reasoning_content": "".join(reasoning) or None,
                    }
                )
            )
    return result


_PARTNER_BLOCKED_OPTIONS = (
    "apiMode",
    "betas",
    "mcp_servers",
    "web_search_options",
    "multi_agent",
    "previous_response_id",
)
_GPT_OSS_EFFORTS = frozenset({"low", "medium", "high"})
_TEMPLATE_EFFORTS = frozenset({"none", "low", "medium", "high"})


def _prepare_partner_options(
    model_id: str,
    profile: _PartnerChatProfile,
    input: ModelGenerateInput,
    options: dict[str, Any],
) -> None:
    """Map shared reasoning onto the Vertex thinking-guide control for this model.

    https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/capabilities/thinking
    """
    for key in _PARTNER_BLOCKED_OPTIONS:
        if options.get(key) is not None:
            raise UnsupportedFeatureError(
                f"Vertex chat does not expose provider_options.{key}."
            )
    reasoning = input.reasoning
    if reasoning is not None and reasoning.budget_tokens is not None:
        raise UnsupportedFeatureError(
            "Vertex partner chat does not expose reasoning budgets, mode or context controls."
        )
    effort = (
        reasoning.effort
        if reasoning is not None and reasoning.effort is not None
        else options.get("reasoning_effort")
    )
    if effort is not None:
        control = profile.thinking
        if control == "effort":
            if effort not in _GPT_OSS_EFFORTS:
                raise UnsupportedFeatureError(
                    "Vertex gpt-oss supports low, medium or high reasoning effort."
                )
            options["reasoning_effort"] = effort
        elif control in {"thinking", "enable_thinking"}:
            if effort not in _TEMPLATE_EFFORTS:
                raise UnsupportedFeatureError(
                    "This Vertex model supports enabling or disabling thinking, not this reasoning effort."
                )
            options.pop("reasoning_effort", None)
            current = options.get("chat_template_kwargs") or {}
            if not isinstance(current, dict):
                raise ValidationError("Vertex chat_template_kwargs must be an object.")
            flag = "thinking" if control == "thinking" else "enable_thinking"
            options["chat_template_kwargs"] = {**current, flag: effort != "none"}
        else:
            raise UnsupportedFeatureError(
                f'Vertex model "{model_id}" does not support reasoning_effort; its thinking behavior is model-defined.'
            )
    if reasoning is not None and not profile.reasoning:
        raise UnsupportedFeatureError(
            f'Vertex model "{model_id}" does not support reasoning.'
        )


def _body(model_id: str, input: ModelGenerateInput, stream: bool) -> dict[str, Any]:
    profile = VERTEX_PARTNER_CHAT_PROFILES.get(model_id)
    if profile is None and input.reasoning is not None:
        raise UnsupportedFeatureError(
            "Use the model's native reasoning options for Vertex Chat Completions."
        )
    options = deepcopy(input.provider_options or {})
    if profile is not None:
        _prepare_partner_options(model_id, profile, input, options)
    mapped_tools = []
    for definition in (input.tools or {}).values():
        if getattr(definition, "kind", None) == "hosted":
            raise UnsupportedFeatureError(
                "Vertex Chat Completions accepts client function tools, not Gemini hosted tools."
            )
        mapped_tools.append(
            {
                "type": "function",
                "function": drop_none(
                    {
                        "name": definition.name,
                        "description": definition.description,
                        "parameters": create_schema_adapter(
                            definition.schema
                        ).json_schema(),
                        "strict": definition.strict,
                    }
                ),
            }
        )
    choice = input.tool_choice
    if profile is not None and profile.thinking == "effort":
        raw_choice = choice if choice is not None else options.get("tool_choice")
        if raw_choice is not None and raw_choice not in ("auto", "none"):
            raise UnsupportedFeatureError(
                "Vertex gpt-oss supports only auto or none tool choice, not required or named tools."
            )
        options.pop("tool_choice", None)
        choice = raw_choice
        for item in mapped_tools:
            item["function"].setdefault("description", "")
    if (
        profile is not None
        and (profile.thinking == "effort" or model_id.startswith("qwen/"))
        and choice is None
        and "tool_choice" not in options
        and mapped_tools
    ):
        choice = "auto"
    response_format = None
    if input.structured_output and input.structured_output.mode == "native":
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": input.structured_output.name or "response",
                "strict": True,
                "schema": create_schema_adapter(
                    input.structured_output.schema
                ).json_schema(),
            },
        }
    body = options
    # SDK-owned fields cannot be redirected through native options.
    body.update(
        drop_none(
            {
                "model": model_id,
                "messages": _messages(input.messages),
                "stream": stream,
                "max_tokens": input.max_tokens,
                "temperature": input.temperature,
                "tools": mapped_tools or None,
                "tool_choice": choice
                if isinstance(choice, str)
                else {"type": "function", "function": {"name": choice.tool_name}}
                if choice
                else None,
                "response_format": response_format,
            }
        )
    )
    if stream:
        body["stream_options"] = {
            **body.get("stream_options", {}),
            "include_usage": True,
        }
    return body


def _usage(payload: dict[str, Any]) -> TokenUsage | None:
    usage = payload.get("usage")
    if not usage:
        return None
    return TokenUsage(
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


def _call(item: dict[str, Any]) -> ToolCall:
    function = item.get("function") or {}
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValidationError(
                "Vertex returned invalid function arguments."
            ) from error
    if not item.get("id") or not function.get("name"):
        raise ValidationError("Vertex returned an incomplete function call.")
    return ToolCall(
        id=item["id"],
        name=function["name"],
        input=arguments,
        provider_metadata={"provider": "vertex", "raw_tool_call": deepcopy(item)},
    )


@dataclass
class VertexChatLanguageModel:
    model_id: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities
    provider: str = "vertex"
    raw_predict: bool = False

    async def _send(self, input: ModelGenerateInput, stream: bool) -> Any:
        validate_message_parts(self, input.messages)
        body = _body(self.model_id, input, stream)
        url = f"{self.base_url}/chat/completions"
        if self.raw_predict:
            publisher, model = self.model_id.split("/", 1)
            body["model"] = model.split("@", 1)[0]
            body.pop("stream_options", None)
            action = "streamRawPredict" if stream else "rawPredict"
            url = f"{self.base_url}/publishers/{publisher}/models/{model}:{action}"

        async def request() -> Any:
            response = await self.fetch(
                url,
                headers={"content-type": "application/json"},
                json_body=body,
                timeout_ms=input.timeout_ms,
                stream=stream,
            )
            if response.status_code >= 400:
                raise ProviderHTTPError(
                    f"Vertex request failed with status {response.status_code}.",
                    response.status_code,
                    response_body=await response.text(),
                )
            return response

        return await with_retry(
            request,
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms
            if input.retry_backoff_ms is not None
            else 250,
        )

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        response = await self._send(input, False)
        payload = await response.json()
        if not payload.get("choices"):
            raise ValidationError("Vertex Chat Completions returned no choices.")
        choice = payload["choices"][0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        parts: list[Any] = [TextPart(text=text)] if text else []
        if message.get("reasoning_content"):
            parts.append(
                ProviderDataPart(
                    provider="vertex",
                    data={"reasoning_content": message["reasoning_content"]},
                )
            )
        parts.extend(
            ToolCallPart(tool_call=_call(item))
            for item in message.get("tool_calls") or []
        )
        return GenerateResult(
            messages=[ModelMessage(role="assistant", parts=parts)],
            text=text,
            finish_reason=normalize_finish_reason(choice.get("finish_reason")),
            provider_finish_reason=choice.get("finish_reason"),
            usage=_usage(payload),
            raw_response=payload,
        )

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        response = await self._send(input, True)

        async def events() -> AsyncIterable[StreamEvent]:
            calls: dict[int, dict[str, Any]] = {}
            finish = None
            usage = None
            lines = response.iter_lines()
            try:
                async for event in parse_sse(lines):
                    if event.data == "[DONE]":
                        break
                    payload = json.loads(event.data)
                    if payload.get("error"):
                        raise ValidationError(
                            "Vertex returned an error during Chat Completions streaming."
                        )
                    usage = _usage(payload) or usage
                    for choice in payload.get("choices") or []:
                        if choice.get("index", 0) != 0:
                            continue
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            yield StreamTextDeltaEvent(text_delta=delta["content"])
                        if delta.get("reasoning_content"):
                            yield StreamProviderDataEvent(
                                provider="vertex",
                                data={"reasoning_content": delta["reasoning_content"]},
                            )
                        for item in delta.get("tool_calls") or []:
                            current = calls.setdefault(
                                item.get("index", 0),
                                {"id": "", "function": {"name": "", "arguments": ""}},
                            )
                            if item.get("id"):
                                current["id"] = item["id"]
                            for key in ("name", "arguments"):
                                current["function"][key] += (
                                    item.get("function") or {}
                                ).get(key) or ""
                        finish = choice.get("finish_reason") or finish
            finally:
                close = getattr(lines, "aclose", None)
                if close is not None:
                    await close()
            if finish is None:
                raise ValidationError(
                    "Vertex Chat Completions stream ended without a finish reason."
                )
            for item in calls.values():
                yield StreamToolCallEvent(tool_call=_call(item))
            yield StreamFinishEvent(
                finish_reason=normalize_finish_reason(finish),
                provider_finish_reason=finish,
                usage=usage,
            )

        return events()
