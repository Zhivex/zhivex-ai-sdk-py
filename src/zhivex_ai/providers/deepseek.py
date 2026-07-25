from __future__ import annotations

from collections.abc import AsyncIterable
from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
import re
from typing import Any
from urllib.parse import urlparse

from .._http import Fetcher, ResponseLike, default_fetch
from .._sse import parse_sse
from ..errors import ConfigurationError, ProviderHTTPError, UnsupportedFeatureError, ValidationError
from ..messages import is_hosted_tool_definition, normalize_finish_reason, validate_message_parts
from ..runtime import with_retry
from ..schema import create_schema_adapter
from ..types import (
    AgentCapabilities,
    GenerateResult,
    LanguageModel,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    PortableSupport,
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
    ToolChoiceName,
    ToolResultPart,
)
from .base import ProviderAdapter, create_provider_bundle


DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_BETA_BASE_URL = "https://api.deepseek.com/beta"
DEEPSEEK_CURRENT_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")
DEEPSEEK_RETIRED_MODELS = frozenset({"deepseek-chat", "deepseek-reasoner"})

DEEPSEEK_AGENT_CAPABILITIES = AgentCapabilities(
    support_tier="tier-b",
    tool_choice_none=True,
)

DEEPSEEK_CHAT_CAPABILITIES = ModelCapabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    parallel_tool_calls=True,
    vision=False,
    files=False,
    audio_input=False,
    audio_output=False,
    embeddings=False,
    reasoning=True,
    web_search=False,
    agent_capabilities=DEEPSEEK_AGENT_CAPABILITIES,
)

_DEEPSEEK_SAMPLING_FIELDS = frozenset(
    {"temperature", "top_p", "presence_penalty", "frequency_penalty"}
)
_DEEPSEEK_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_DEEPSEEK_STRICT_TYPES = frozenset(
    {"object", "string", "number", "integer", "boolean", "array"}
)
_RESERVED_PROVIDER_OPTIONS = frozenset({"model", "messages", "tools", "stream", "max_tokens"})


def _headers(api_key: str) -> dict[str, str]:
    return {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }


def _is_beta_base_url(base_url: str) -> bool:
    return urlparse(base_url).path.rstrip("/").endswith("/beta")


def _uses_beta_features(input: ModelGenerateInput) -> bool:
    if any(getattr(definition, "strict", False) is True for definition in (input.tools or {}).values()):
        return True
    return any(
        isinstance(part, ProviderDataPart)
        and part.provider == "deepseek"
        and isinstance(part.data, dict)
        and part.data.get("prefix") is True
        for message in input.messages
        for part in message.parts
    )


def _request_base_url(base_url: str, input: ModelGenerateInput) -> str:
    if not _uses_beta_features(input) or _is_beta_base_url(base_url):
        return base_url
    parsed = urlparse(base_url)
    if parsed.hostname == "api.deepseek.com" and parsed.path.rstrip("/") in {"", "/v1"}:
        return DEEPSEEK_BETA_BASE_URL
    return f"{base_url.rstrip('/')}/beta"


def _tool_result_content(part: ToolResultPart) -> str:
    result = part.tool_result
    if result.is_error and result.error is not None:
        return json.dumps({"message": result.error.message})
    return json.dumps(result.output)


def _provider_data(message: ModelMessage, key: str) -> Any:
    values: list[Any] = []
    for part in message.parts:
        if (
            isinstance(part, ProviderDataPart)
            and part.provider == "deepseek"
            and isinstance(part.data, dict)
            and key in part.data
        ):
            values.append(part.data[key])
    if not values:
        return None
    if all(isinstance(value, str) for value in values):
        return "".join(values)
    return deepcopy(values[0])


def _message_text(message: ModelMessage) -> str | None:
    chunks = [part.text for part in message.parts if isinstance(part, TextPart)]
    return "".join(chunks) if chunks else None


def _to_chat_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            for part in message.parts:
                if isinstance(part, ToolResultPart):
                    mapped.append(
                        {
                            "role": "tool",
                            "tool_call_id": part.tool_result.tool_call_id,
                            "content": _tool_result_content(part),
                        }
                    )
            continue

        payload: dict[str, Any] = {"role": message.role}
        content = _message_text(message)
        if content is not None:
            payload["content"] = content

        tool_calls = [part.tool_call for part in message.parts if isinstance(part, ToolCallPart)]
        if tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.input),
                    },
                }
                for call in tool_calls
            ]
            # DeepSeek V4 requires a non-null assistant content field on tool-call
            # turns, even when there is no user-facing text.
            payload.setdefault("content", "")

        reasoning_content = _provider_data(message, "reasoning_content")
        if reasoning_content is not None:
            payload["reasoning_content"] = reasoning_content

        prefix = _provider_data(message, "prefix")
        if prefix is not None:
            payload["prefix"] = bool(prefix)

        if "content" in payload or "tool_calls" in payload:
            mapped.append(payload)
    return mapped


def _validate_strict_schema(tool_name: str, schema: dict[str, Any]) -> None:
    errors: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return

        node_type = node.get("type")
        if isinstance(node_type, str) and node_type not in _DEEPSEEK_STRICT_TYPES:
            errors.append(f'unsupported JSON Schema type "{node_type}"')
        if node_type == "object":
            properties = node.get("properties")
            if not isinstance(properties, dict):
                properties = {}
            required = node.get("required")
            required_names = set(required) if isinstance(required, list) else set()
            missing = sorted(name for name in properties if name not in required_names)
            if missing:
                errors.append(f'mark every property as required (missing: {", ".join(missing)})')
            if node.get("additionalProperties") is not False:
                errors.append('set "additionalProperties": false on every object')

        for value in node.values():
            visit(value)

    visit(schema)
    if errors:
        unique_errors = list(dict.fromkeys(errors))
        raise ValidationError(
            f'DeepSeek strict tool "{tool_name}" uses an incompatible schema: '
            f'{"; ".join(unique_errors)}.'
        )


def _map_tools(
    tools: dict[str, Any] | None,
    *,
    base_url: str,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    if len(tools) > 128:
        raise ValidationError("DeepSeek accepts at most 128 function tools per request.")

    strict_values: list[bool] = []
    mapped: list[dict[str, Any]] = []
    for definition in tools.values():
        if is_hosted_tool_definition(definition):
            raise UnsupportedFeatureError('Provider "deepseek" only supports callable function tools.')
        if not _DEEPSEEK_TOOL_NAME.fullmatch(definition.name):
            raise ValidationError(
                f'DeepSeek tool name "{definition.name}" must contain only letters, digits, "_", or "-", '
                "with a maximum length of 64 characters."
            )
        parameters = create_schema_adapter(definition.schema).json_schema()
        function = {
            "name": definition.name,
            "description": definition.description,
            "parameters": parameters,
        }
        if definition.strict is not None:
            function["strict"] = definition.strict
            strict_values.append(definition.strict)
        if definition.strict:
            if not _is_beta_base_url(base_url):
                raise UnsupportedFeatureError(
                    'DeepSeek strict tool mode is beta. Create the provider with '
                    f'base_url="{DEEPSEEK_BETA_BASE_URL}" before using tool(..., strict=True).'
                )
            _validate_strict_schema(definition.name, parameters)
        mapped.append({"type": "function", "function": function})

    strict_enabled = any(strict_values)
    if strict_enabled and (
        not all(strict_values) or len(strict_values) != len(mapped)
    ):
        raise ValidationError("DeepSeek strict mode requires every function tool to set strict=True.")
    return mapped


def _map_tool_choice(tool_choice: str | ToolChoiceName | None) -> str | dict[str, Any] | None:
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    return {
        "type": "function",
        "function": {"name": tool_choice.tool_name},
    }


def _normalize_reasoning_effort(value: Any) -> str:
    if value in {"low", "medium", "high"}:
        return "high"
    if value in {"xhigh", "max"}:
        return "max"
    raise UnsupportedFeatureError(
        f'DeepSeek reasoning effort must be low, medium, high, xhigh, or max; received "{value}".'
    )


def _validate_thinking_option(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or value.get("type") not in {"enabled", "disabled"}:
        raise ValidationError(
            'DeepSeek provider option "thinking" must be {"type": "enabled"} or {"type": "disabled"}.'
        )
    return {"type": str(value["type"])}


def _resolve_thinking(
    input: ModelGenerateInput,
    provider_options: dict[str, Any],
) -> tuple[dict[str, str] | None, str | None, bool]:
    option_thinking = provider_options.pop("thinking", None)
    option_effort = provider_options.pop("reasoning_effort", None)
    option_tool_choice = provider_options.get("tool_choice")
    option_sampling = _DEEPSEEK_SAMPLING_FIELDS.intersection(provider_options)

    if input.reasoning is not None:
        if option_thinking is not None or option_effort is not None:
            raise ValidationError(
                'Pass either reasoning=... or DeepSeek provider_options["thinking"]/'
                'provider_options["reasoning_effort"], not both.'
            )
        if input.reasoning.budget_tokens is not None:
            raise UnsupportedFeatureError('Provider "deepseek" does not support "reasoning.budgetTokens".')
        if input.reasoning.effort == "none":
            return {"type": "disabled"}, None, False
        if input.reasoning.effort in {None, "minimal"}:
            raise UnsupportedFeatureError(
                'Provider "deepseek" does not support reasoning effort "minimal"; '
                "use low, medium, high, xhigh, max, or none."
            )
        effort = _normalize_reasoning_effort(input.reasoning.effort)
        return {"type": "enabled"}, effort, True

    if option_thinking is not None:
        thinking = _validate_thinking_option(option_thinking)
        enabled = thinking["type"] == "enabled"
        if option_effort is not None and not enabled:
            raise ValidationError(
                'DeepSeek provider option "reasoning_effort" cannot be combined with thinking disabled.'
            )
        configured_effort = (
            _normalize_reasoning_effort(option_effort)
            if option_effort is not None
            else None
        )
        return thinking, configured_effort, enabled

    if option_effort is not None:
        return {"type": "enabled"}, _normalize_reasoning_effort(option_effort), True

    # V4 defaults to thinking. Disable it implicitly when the caller requests a
    # setting that DeepSeek documents as incompatible with thinking mode, so the
    # portable temperature and tool-choice contracts remain effective.
    if (
        input.temperature is not None
        or input.tool_choice is not None
        or option_tool_choice is not None
        or bool(option_sampling)
    ):
        return {"type": "disabled"}, None, False
    return None, None, True


def _structured_output_instruction(input: ModelGenerateInput) -> str | None:
    structured = input.structured_output
    if structured is None or structured.mode != "native":
        return None
    schema = create_schema_adapter(structured.schema).json_schema()
    return (
        "Return only valid JSON. The JSON value must match this JSON Schema exactly:\n"
        f"{json.dumps(schema, separators=(',', ':'), ensure_ascii=False)}"
    )


def _validate_prefix_messages(messages: list[dict[str, Any]], *, base_url: str) -> None:
    prefixes = [index for index, message in enumerate(messages) if message.get("prefix") is True]
    if not prefixes:
        return
    if not _is_beta_base_url(base_url):
        raise UnsupportedFeatureError(
            'DeepSeek Chat Prefix Completion is beta. Create the provider with '
            f'base_url="{DEEPSEEK_BETA_BASE_URL}".'
        )
    if prefixes != [len(messages) - 1] or messages[-1].get("role") != "assistant":
        raise ValidationError(
            "DeepSeek Chat Prefix Completion requires prefix=True only on the final assistant message."
        )


def _chat_body(
    model_id: str,
    base_url: str,
    input: ModelGenerateInput,
    *,
    stream: bool,
) -> dict[str, Any]:
    if model_id.strip().lower() in DEEPSEEK_RETIRED_MODELS:
        current = ", ".join(DEEPSEEK_CURRENT_MODELS)
        raise UnsupportedFeatureError(
            f'DeepSeek model "{model_id}" was retired on 2026-07-24. Use one of: {current}.'
        )

    provider_options = deepcopy(input.provider_options or {})
    reserved = sorted(_RESERVED_PROVIDER_OPTIONS.intersection(provider_options))
    if reserved:
        raise ValidationError(
            "DeepSeek provider_options cannot override SDK-owned fields: "
            + ", ".join(reserved)
            + "."
        )
    if input.temperature is not None and "temperature" in provider_options:
        raise ValidationError('Pass either temperature=... or provider_options={"temperature": ...}, not both.')
    if input.tool_choice is not None and "tool_choice" in provider_options:
        raise ValidationError('Pass either tool_choice=... or provider_options={"tool_choice": ...}, not both.')

    thinking, reasoning_effort, thinking_enabled = _resolve_thinking(input, provider_options)
    if thinking_enabled:
        incompatible = sorted(_DEEPSEEK_SAMPLING_FIELDS.intersection(provider_options))
        if input.temperature is not None:
            incompatible.insert(0, "temperature")
        if incompatible:
            raise UnsupportedFeatureError(
                "DeepSeek thinking mode ignores sampling fields; disable thinking before setting "
                + ", ".join(dict.fromkeys(incompatible))
                + "."
            )
        if input.tool_choice is not None or provider_options.get("tool_choice") is not None:
            raise UnsupportedFeatureError(
                "DeepSeek V4 does not accept tool_choice while thinking is enabled. "
                'Use reasoning=ReasoningConfig(effort="none") or omit tool_choice.'
            )

    messages = _to_chat_messages(input.messages)
    structured_instruction = _structured_output_instruction(input)
    if structured_instruction is not None:
        messages.insert(0, {"role": "system", "content": structured_instruction})
    _validate_prefix_messages(messages, base_url=base_url)

    mapped_tools = _map_tools(input.tools, base_url=base_url)
    stream_options = deepcopy(provider_options.pop("stream_options", None))
    if stream:
        if stream_options is None:
            stream_options = {}
        if not isinstance(stream_options, dict):
            raise ValidationError('DeepSeek provider option "stream_options" must be an object.')
        stream_options.setdefault("include_usage", True)

    body = {
        "model": model_id,
        "messages": messages,
        "tools": mapped_tools,
        "tool_choice": _map_tool_choice(input.tool_choice),
        "temperature": input.temperature,
        "max_tokens": input.max_tokens,
        "response_format": (
            {"type": "json_object"}
            if input.structured_output is not None and input.structured_output.mode == "native"
            else None
        ),
        "thinking": thinking,
        "reasoning_effort": reasoning_effort,
        **provider_options,
        "stream": True if stream else None,
        "stream_options": stream_options,
    }
    return {key: value for key, value in body.items() if value is not None}


def _parse_usage(payload: dict[str, Any]) -> TokenUsage | None:
    usage = payload.get("usage") or {}
    if not usage:
        return None
    return TokenUsage(
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


def _parse_tool_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return deepcopy(value) if value is not None else {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_tool_call(value: dict[str, Any]) -> ToolCall:
    function = value.get("function") or {}
    return ToolCall(
        id=str(value.get("id") or ""),
        name=str(function.get("name") or ""),
        input=_parse_tool_arguments(function.get("arguments")),
        provider_metadata={
            "provider": "deepseek",
            "raw_tool_call": deepcopy(value),
        },
    )


def _parse_message(payload: dict[str, Any], usage: dict[str, Any] | None = None) -> ModelMessage:
    parts: list[Any] = []
    content = payload.get("content")
    if isinstance(content, str) and content:
        parts.append(TextPart(text=content))
    reasoning_content = payload.get("reasoning_content")
    if reasoning_content is not None:
        parts.append(
            ProviderDataPart(
                provider="deepseek",
                data={"reasoning_content": reasoning_content},
            )
        )
    if usage:
        parts.append(
            ProviderDataPart(
                provider="deepseek",
                data={"usage": deepcopy(usage)},
            )
        )
    for item in payload.get("tool_calls") or []:
        if isinstance(item, dict):
            parts.append(ToolCallPart(tool_call=_parse_tool_call(item)))
    return ModelMessage(role="assistant", parts=parts)


def _provider_error(response: ResponseLike, body: str) -> ProviderHTTPError:
    return ProviderHTTPError(
        f"deepseek request failed with status {response.status_code}.",
        response.status_code,
        response_body=body,
        response_headers=dict(getattr(response, "headers", {}) or {}),
    )


async def _checked_fetch(
    *,
    fetch: Fetcher,
    url: str,
    headers: dict[str, str],
    json_body: dict[str, Any],
    timeout_ms: int | None,
    stream: bool,
) -> ResponseLike:
    response = await fetch(
        url,
        headers=headers,
        json_body=json_body,
        timeout_ms=timeout_ms,
        stream=stream,
    )
    if response.status_code >= 400:
        raise _provider_error(response, await response.text())
    return response


@dataclass(slots=True)
class DeepSeekChatLanguageModel(LanguageModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: DEEPSEEK_CHAT_CAPABILITIES)

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        validate_message_parts(self, input.messages)
        request_base_url = _request_base_url(self.base_url, input)
        body = _chat_body(self.model_id, request_base_url, input, stream=False)

        async def request() -> dict[str, Any]:
            response = await _checked_fetch(
                fetch=self.fetch,
                url=f"{request_base_url}/chat/completions",
                headers=_headers(self.api_key),
                json_body=body,
                timeout_ms=input.timeout_ms,
                stream=False,
            )
            payload = await response.json()
            choice = ((payload.get("choices") or [{}])[0] or {})
            if choice.get("finish_reason") == "insufficient_system_resource":
                raise ProviderHTTPError(
                    "deepseek request ended because inference resources were unavailable.",
                    503,
                    response_body=json.dumps(payload),
                    response_headers=dict(getattr(response, "headers", {}) or {}),
                    retryable=True,
                )
            return payload

        payload = await with_retry(
            request,
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        choice = ((payload.get("choices") or [{}])[0] or {})
        message_payload = choice.get("message") or {}
        usage_payload = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
        message = _parse_message(message_payload, usage_payload)
        finish = choice.get("finish_reason")
        return GenerateResult(
            messages=[message],
            text="".join(part.text for part in message.parts if isinstance(part, TextPart)),
            finish_reason=normalize_finish_reason(finish),
            provider_finish_reason=finish,
            usage=_parse_usage(payload),
            raw_response=payload,
        )

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        validate_message_parts(self, input.messages)
        request_base_url = _request_base_url(self.base_url, input)
        body = _chat_body(self.model_id, request_base_url, input, stream=True)
        response = await with_retry(
            lambda: _checked_fetch(
                fetch=self.fetch,
                url=f"{request_base_url}/chat/completions",
                headers=_headers(self.api_key),
                json_body=body,
                timeout_ms=input.timeout_ms,
                stream=True,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )

        async def generator() -> AsyncIterable[StreamEvent]:
            tool_call_accumulator: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None
            usage_payload: dict[str, Any] | None = None

            async def terminal_events() -> AsyncIterable[StreamEvent]:
                for current in tool_call_accumulator.values():
                    yield StreamToolCallEvent(tool_call=_parse_tool_call(current))
                if usage_payload:
                    yield StreamProviderDataEvent(
                        provider="deepseek",
                        data={"usage": deepcopy(usage_payload)},
                    )
                if finish_reason == "insufficient_system_resource":
                    yield StreamFinishEvent(
                        finish_reason="error",
                        provider_finish_reason=finish_reason,
                        usage=_parse_usage({"usage": usage_payload or {}}),
                    )
                elif finish_reason is not None or usage_payload is not None:
                    yield StreamFinishEvent(
                        finish_reason=normalize_finish_reason(finish_reason),
                        provider_finish_reason=finish_reason,
                        usage=_parse_usage({"usage": usage_payload or {}}),
                    )

            async for event in parse_sse(response.iter_lines()):
                if event.data == "[DONE]":
                    async for terminal in terminal_events():
                        yield terminal
                    return
                payload = json.loads(event.data)
                if isinstance(payload.get("usage"), dict):
                    usage_payload = dict(payload["usage"])
                choices = payload.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] or {}
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    yield StreamTextDeltaEvent(text_delta=str(delta.get("content") or ""))
                if delta.get("reasoning_content") is not None:
                    yield StreamProviderDataEvent(
                        provider="deepseek",
                        data={"reasoning_content": delta.get("reasoning_content")},
                    )
                for item in delta.get("tool_calls") or []:
                    index = int(item.get("index") or 0)
                    current = tool_call_accumulator.setdefault(
                        index,
                        {"id": "", "function": {"name": "", "arguments": ""}},
                    )
                    if item.get("id"):
                        current["id"] = item["id"]
                    function = item.get("function") or {}
                    current_function = current.setdefault(
                        "function",
                        {"name": "", "arguments": ""},
                    )
                    if function.get("name"):
                        current_function["name"] = function["name"]
                    if function.get("arguments"):
                        current_function["arguments"] = (
                            str(current_function.get("arguments") or "")
                            + str(function.get("arguments") or "")
                        )
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])

            async for terminal in terminal_events():
                yield terminal

        return generator()


def create_deepseek(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    fetch: Fetcher | None = None,
):
    resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not resolved_key:
        raise ConfigurationError("Missing deepseek API key. Set DEEPSEEK_API_KEY.")
    resolved_base_url = (
        base_url
        or os.getenv("DEEPSEEK_BASE_URL")
        or DEEPSEEK_DEFAULT_BASE_URL
    ).rstrip("/")
    requester = fetch or default_fetch

    native = ProviderAdapter(
        name="deepseek",
        language_model_factory=lambda model_id: DeepSeekChatLanguageModel(
            provider="deepseek",
            model_id=model_id,
            api_key=resolved_key,
            base_url=resolved_base_url,
            fetch=requester,
        ),
    )
    return create_provider_bundle(
        name="deepseek",
        native=native,
        agent_capabilities=DEEPSEEK_AGENT_CAPABILITIES,
        portable_support=PortableSupport(
            text_generation=True,
            streaming=True,
            structured_output=True,
            tools=True,
            embeddings=False,
            grounding=False,
            retrieval=False,
            transcription=False,
            speech=False,
            portable_badge=True,
            tier="portable",
        ),
    )
