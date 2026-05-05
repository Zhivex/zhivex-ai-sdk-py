from __future__ import annotations

import asyncio
import fnmatch
import inspect
import json
import re
import sqlite3
import time
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, cast
from uuid import uuid4

from ._http import default_fetch
from ._serde import (
    deserialize_generate_result,
    deserialize_messages,
    deserialize_model_generate_input,
    serialize_mcp_server_config,
    serialize_generate_result,
    serialize_messages,
    serialize_model_generate_input,
    serialize_tool_execution_context,
)
from .errors import ProviderHTTPError, ValidationError
from .agent_state import (
    AgentChildRun,
    AgentRunState,
    AgentRunStatus,
    AgentRunStep,
    AgentRunStore,
    agent_child_run_from_state,
)
from .generate_text import generate_text, stream_text
from .messages import create_text_message, is_callable_tool_definition, provider_data_part, tool_result_part
from .skills import SkillArtifact, SkillDefinition, SkillRegistry, SkillSet
from .types import (
    AnyToolDefinition,
    AudioFrame,
    FinishReason,
    GenerateTextOutput,
    GenerateTextStep,
    JsonValue,
    LanguageModel,
    MCPServerConfig,
    MCPToolConfig,
    ModelGenerateInput,
    ModelMessage,
    ReasoningConfig,
    RemoteHTTPToolConfig,
    RealtimeConnectOptions,
    RealtimeEvent,
    RealtimeModel,
    RealtimeResponseCompletedEvent,
    RealtimeSessionConfig,
    RealtimeSessionEndedEvent,
    RealtimeTextDeltaEvent,
    RealtimeToolCallEvent,
    RealtimeToolResultEvent,
    RealtimeTranscriptEvent,
    StreamProviderDataEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    StreamToolResultEvent,
    ToolChoiceName,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionError,
    ToolExecutionOptions,
    ToolExecutionResult,
    ToolSet,
    TokenUsage,
    ToolCall,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

HANDOFF_MARKER = "__zhivex_agent_handoff__"
SUMMARY_MARKER = "Conversation summary:\n"
_POSTGRES_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _text_from_message(message: ModelMessage) -> str:
    return "".join(part.text for part in message.parts if isinstance(part, TextPart))


def _message_text(messages: Iterable[ModelMessage]) -> str:
    chunks: list[str] = []
    for message in messages:
        text = _text_from_message(message).strip()
        if text:
            chunks.append(f"{message.role}: {text}")
    return "\n".join(chunks)


def _skill_reference_name(skill: SkillDefinition) -> str:
    return skill.display_name or skill.name


SkillActivationMode = Literal["explicit", "implicit", "sticky"]


@dataclass(slots=True)
class _SkillActivation:
    skill: SkillDefinition
    mode: SkillActivationMode


@dataclass(slots=True)
class _SkillSkip:
    skill_name: str
    reason: str
    mode: SkillActivationMode
    path: str | None = None


def _normalize_skill_names(values: Iterable[Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _tokenize_skill_text(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}", value.lower()) if len(token) >= 3}


def _matches_any_phrase(text: str, patterns: Iterable[str]) -> bool:
    lowered = text.lower()
    for pattern in patterns:
        candidate = pattern.strip().lower()
        if candidate and candidate in lowered:
            return True
    return False


def _explicit_skill_requested(skill: SkillDefinition, text: str) -> bool:
    pattern = re.compile(rf"(?<!\w)\${re.escape(skill.name)}(?![\w-])", re.IGNORECASE)
    return bool(pattern.search(text))


def _should_activate_skill_implicitly(skill: SkillDefinition, text: str) -> bool:
    if not skill.allow_implicit_invocation:
        return False
    trigger_matched = not skill.triggers or _matches_any_phrase(text, skill.triggers)
    if not trigger_matched:
        return False
    if skill.anti_triggers and _matches_any_phrase(text, skill.anti_triggers):
        return False
    if skill.triggers:
        return True
    lowered = text.lower()
    if skill.name.lower() in lowered:
        return True
    prompt_tokens = _tokenize_skill_text(text)
    if not prompt_tokens:
        return False
    name_tokens = _tokenize_skill_text(skill.name.replace("-", " "))
    description_tokens = _tokenize_skill_text(skill.description)
    overlap = len(prompt_tokens & (name_tokens | description_tokens))
    return overlap >= 2 or (overlap >= 1 and len(name_tokens) == 1 and bool(prompt_tokens & name_tokens))


def _skill_allowed_for_agent(skill: SkillDefinition, agent: Agent) -> tuple[bool, str | None]:
    provider = str(getattr(agent.model, "provider", "") or "")
    model_id = str(getattr(agent.model, "model_id", "") or "")
    if skill.allowed_providers and provider not in skill.allowed_providers:
        return False, f'provider "{provider}" is not allowed'
    if skill.allowed_models and not any(fnmatch.fnmatch(model_id, pattern) for pattern in skill.allowed_models):
        return False, f'model "{model_id}" is not allowed'
    return True, None


def _skill_activation_sort_key(item: _SkillActivation) -> tuple[int, int, str]:
    mode_order = {"explicit": 0, "sticky": 1, "implicit": 2}
    return (-item.skill.priority, mode_order[item.mode], item.skill.name)


def _skill_system_message(skill: SkillDefinition) -> str:
    lines = [
        f"[Active skill: {_skill_reference_name(skill)}]",
        f"Description: {skill.description}",
    ]
    if skill.version:
        lines.append(f"Version: {skill.version}")
    if skill.path:
        lines.append(f"Skill file: {skill.path}")
        skill_dir = str(Path(skill.path).resolve().parent)
        lines.append(f"Skill directory: {skill_dir}")
        for label, directory in (
            ("scripts", Path(skill_dir) / "scripts"),
            ("references", Path(skill_dir) / "references"),
            ("assets", Path(skill_dir) / "assets"),
        ):
            if directory.exists():
                lines.append(f"Available {label}: {directory}")
    if skill.resources:
        lines.append(f"Available resources: {', '.join(skill.resources)}")
    if skill.entrypoints:
        lines.append(f"Available entrypoints: {', '.join(item.name for item in skill.entrypoints)}")
    if skill.default_prompt:
        lines.append(f"Suggested surrounding prompt: {skill.default_prompt}")
    lines.append("Follow these skill instructions for the current task:")
    lines.append(skill.instructions.strip())
    return "\n".join(lines).strip()


def _strip_runtime_system_messages(messages: list[ModelMessage], instructions: str | None) -> list[ModelMessage]:
    stripped = list(messages)
    while stripped and stripped[0].role == "system":
        text = _text_from_message(stripped[0])
        if instructions and text == instructions:
            stripped.pop(0)
            continue
        if text.startswith(SUMMARY_MARKER):
            stripped.pop(0)
            continue
        break
    return stripped


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


def _invoke_tool_callable(execute: Any, parsed: Any, context: ToolExecutionContext) -> Any:
    mode = _tool_callable_mode(execute)
    if mode == "kwargs":
        return execute(parsed, context=context)
    if mode == "positional":
        return execute(parsed, context)
    return execute(parsed)


@lru_cache(maxsize=256)
def _tool_callable_mode(execute: Any) -> str:
    try:
        signature = inspect.signature(execute)
    except (TypeError, ValueError):
        return "single"
    parameters = list(signature.parameters.values())
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return "kwargs"
    if any(parameter.name == "context" for parameter in parameters):
        return "kwargs"
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2:
        return "positional"
    return "single"


def handoff_to(target_agent: str, *, input: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        HANDOFF_MARKER: True,
        "target_agent": target_agent,
        "input": input,
        "metadata": metadata or {},
    }


@dataclass(slots=True)
class AgentHandoff:
    target_agent: str
    input: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentContext:
    run_id: str
    session_id: str
    agent_name: str
    memory_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    handoff_path: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ApprovalDecision:
    approved: bool
    reason: str | None = None


@dataclass(slots=True)
class ToolApprovalRequest:
    run_id: str
    session_id: str
    agent_name: str
    tool_name: str
    tool_input: Any
    tool_permissions: list[str] = field(default_factory=list)
    tool_source: str = "local"
    tool_metadata: dict[str, Any] = field(default_factory=dict)
    context: AgentContext | None = None
    handoff_path: list[str] = field(default_factory=list)


class ApprovalPolicy(Protocol):
    async def __call__(self, request: ToolApprovalRequest) -> ApprovalDecision | bool: ...


@dataclass(slots=True)
class GuardrailResult:
    tripwire_triggered: bool = False
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InputGuardrailRequest:
    run_id: str
    session_id: str
    agent_name: str
    prompt: str | None = None
    messages: list[ModelMessage] = field(default_factory=list)
    context: AgentContext | None = None


@dataclass(slots=True)
class OutputGuardrailRequest:
    run_id: str
    session_id: str
    agent_name: str
    text: str = ""
    messages: list[ModelMessage] = field(default_factory=list)
    result: GenerateTextOutput | None = None
    context: AgentContext | None = None


class InputGuardrail(Protocol):
    async def __call__(self, request: InputGuardrailRequest) -> GuardrailResult | bool: ...


class OutputGuardrail(Protocol):
    async def __call__(self, request: OutputGuardrailRequest) -> GuardrailResult | bool: ...


class GuardrailTripwireTriggered(RuntimeError):
    def __init__(
        self,
        *,
        stage: Literal["input", "output"],
        guardrail_name: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.stage = stage
        self.guardrail_name = guardrail_name
        self.reason = reason
        self.metadata = dict(metadata or {})
        message = f'Agent {stage} guardrail "{guardrail_name}" triggered.'
        if reason:
            message = f"{message} {reason}"
        super().__init__(message)


@dataclass(slots=True)
class SummaryConfig:
    max_messages: int = 12
    preserve_recent_messages: int = 8
    max_summary_chars: int = 1200


@dataclass(slots=True)
class AgentMemoryState:
    messages: list[ModelMessage] = field(default_factory=list)
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentMemory(Protocol):
    summary_config: SummaryConfig

    async def load(self, session_id: str) -> AgentMemoryState: ...

    async def save(self, session_id: str, state: AgentMemoryState) -> None: ...

    async def summarize(
        self,
        *,
        session_id: str,
        state: AgentMemoryState,
        agent: "Agent",
    ) -> str | None: ...


class AgentCheckpointStore(Protocol):
    async def save(self, checkpoint: "AgentCheckpoint") -> None: ...

    async def get_latest(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> "AgentCheckpoint | None": ...

    async def list(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> list["AgentCheckpoint"]: ...


@dataclass(slots=True)
class RunLimits:
    max_steps: int | None = 8
    max_tool_calls: int | None = 32
    max_wall_time_ms: int | None = 120_000
    max_handoffs: int | None = 1


@dataclass(slots=True)
class AgentSession:
    id: str = field(default_factory=lambda: _new_id("session"))
    messages: list[ModelMessage] = field(default_factory=list)
    summary: str | None = None
    state: dict[str, JsonValue] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRuntime(Protocol):
    async def execute(self, definition: AnyToolDefinition, input: Any, context: ToolExecutionContext) -> Any: ...

    async def aclose(self) -> None: ...


class LocalToolRuntime:
    async def execute(self, definition: AnyToolDefinition, input: Any, context: ToolExecutionContext) -> Any:
        if not is_callable_tool_definition(definition):
            raise RuntimeError(f'Tool "{definition.name}" is provider-managed and cannot run in the local tool runtime.')
        if definition.execute is None:
            raise RuntimeError(f'Tool "{definition.name}" does not define a local executor.')
        result = _invoke_tool_callable(definition.execute, input, context)
        return await _maybe_await(result)

    async def aclose(self) -> None:
        return None


class UnsupportedToolRuntime:
    def __init__(self, source: str) -> None:
        self._source = source

    async def execute(self, definition: AnyToolDefinition, input: Any, context: ToolExecutionContext) -> Any:
        raise RuntimeError(
            f'Tool "{definition.name}" uses source "{self._source}", but no runtime is configured for that source.'
        )

    async def aclose(self) -> None:
        return None


class HTTPRemoteToolRuntime:
    def __init__(self, *, fetch: Any = None) -> None:
        self._fetch = fetch or default_fetch

    async def execute(self, definition: AnyToolDefinition, input: Any, context: ToolExecutionContext) -> Any:
        if not is_callable_tool_definition(definition):
            raise RuntimeError(f'Tool "{definition.name}" is provider-managed and cannot run in the remote tool runtime.')
        config: RemoteHTTPToolConfig | None = definition.remote_config
        if config is None:
            raise RuntimeError(f'Tool "{definition.name}" does not define a remote_config.')
        response = await self._fetch(
            config.url,
            method="POST",
            headers={"content-type": "application/json", **dict(config.headers)},
            json_body={
                "tool": definition.name,
                "input": input,
                "context": serialize_tool_execution_context(context),
            },
            timeout_ms=config.timeout_ms,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f'Remote tool "{definition.name}" failed with status {response.status_code}.',
                response.status_code,
                response_body=await response.text(),
            )
        payload = await response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f'Remote tool "{definition.name}" returned a non-object payload.')
        if isinstance(payload.get("error"), dict):
            raise RuntimeError(str(payload["error"].get("message") or f'Remote tool "{definition.name}" failed.'))
        if "output" not in payload:
            raise RuntimeError(f'Remote tool "{definition.name}" response must include an "output" field.')
        return payload["output"]

    async def aclose(self) -> None:
        return None


def _normalize_mcp_content_item(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_mcp_content_item(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_mcp_content_item(item) for item in value]
    if is_dataclass(value):
        return _normalize_mcp_content_item(asdict(cast("DataclassInstance", value)))
    return value


def _normalize_mcp_result(payload: Any) -> Any:
    if isinstance(payload, dict):
        if "structuredContent" in payload:
            return payload["structuredContent"]
        if "structured_content" in payload:
            return payload["structured_content"]
        if "content" in payload:
            content = payload["content"]
        else:
            content = None
    else:
        content = getattr(payload, "content", None)
        structured = getattr(payload, "structuredContent", None)
        if structured is None:
            structured = getattr(payload, "structured_content", None)
        if structured is not None:
            return _normalize_mcp_content_item(structured)
    if content is None:
        return _normalize_mcp_content_item(payload)
    normalized: list[Any] = []
    for item in content or []:
        item_payload = _normalize_mcp_content_item(item)
        if isinstance(item_payload, dict) and item_payload.get("type") == "text":
            normalized.append(item_payload.get("text", ""))
        else:
            normalized.append(item_payload)
    if len(normalized) == 1:
        return normalized[0]
    return normalized


class MCPToolRuntime:
    def __init__(self) -> None:
        self._sessions: dict[str, tuple[Any, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _key(self, server: MCPServerConfig) -> str:
        return _json_dumps(serialize_mcp_server_config(server))

    async def _load_client_api(self) -> tuple[Any, Any, Any]:
        try:
            from mcp import ClientSession  # type: ignore[import-not-found]
            from mcp.client.stdio import StdioServerParameters, stdio_client  # type: ignore[import-not-found]
            from mcp.client.streamable_http import streamable_http_client  # type: ignore[import-not-found]
        except Exception as error:
            raise RuntimeError('MCP support requires the optional dependency "mcp".') from error
        return ClientSession, (StdioServerParameters, stdio_client), streamable_http_client

    async def _get_session(self, server: MCPServerConfig) -> Any:
        key = self._key(server)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._sessions:
                return self._sessions[key][1]
            ClientSession, stdio_bundle, streamable_http_client = await self._load_client_api()
            if server.transport == "stdio":
                StdioServerParameters, stdio_client = stdio_bundle
                if not server.command:
                    raise ValidationError('MCP stdio servers require "command".')
                transport_cm = stdio_client(
                    StdioServerParameters(
                        command=server.command,
                        args=list(server.args),
                        env=dict(server.env) or None,
                    )
                )
            elif server.transport == "streamable-http":
                if not server.url:
                    raise ValidationError('MCP streamable-http servers require "url".')
                transport_cm = streamable_http_client(
                    server.url,
                    headers=dict(server.headers) or None,
                    timeout=server.timeout_ms / 1000 if server.timeout_ms is not None else None,
                )
            else:
                raise ValidationError(f'Unsupported MCP transport "{server.transport}".')

            transport = await transport_cm.__aenter__()
            if not isinstance(transport, tuple) or len(transport) != 2:
                raise RuntimeError("MCP transport client did not return the expected read/write streams.")
            read_stream, write_stream = transport
            session_cm = ClientSession(read_stream, write_stream)
            session = await session_cm.__aenter__()
            await session.initialize()
            self._sessions[key] = ((transport_cm, session_cm), session)
            return session

    async def list_tools(self, server: MCPServerConfig) -> list[Any]:
        session = await self._get_session(server)
        result = await session.list_tools()
        tools = getattr(result, "tools", None)
        if tools is None and isinstance(result, dict):
            tools = result.get("tools")
        return list(tools or [])

    async def execute(self, definition: AnyToolDefinition, input: Any, context: ToolExecutionContext) -> Any:
        if not is_callable_tool_definition(definition):
            raise RuntimeError(f'Tool "{definition.name}" is provider-managed and cannot run in the MCP tool runtime.')
        config: MCPToolConfig | None = definition.mcp_config
        if config is None:
            raise RuntimeError(f'Tool "{definition.name}" does not define an mcp_config.')
        session = await self._get_session(config.server)
        result = await session.call_tool(config.tool_name, arguments=input)
        return _normalize_mcp_result(result)

    async def aclose(self) -> None:
        for managers, _session in list(self._sessions.values()):
            transport_cm, session_cm = managers
            await session_cm.__aexit__(None, None, None)
            await transport_cm.__aexit__(None, None, None)
        self._sessions.clear()
        self._locks.clear()


class ToolRegistry:
    def __init__(
        self,
        tools: ToolSet | None = None,
        *,
        runtimes: dict[str, ToolRuntime] | None = None,
    ) -> None:
        self._tools: ToolSet = dict(tools or {})
        self._runtimes: dict[str, ToolRuntime] = {
            "local": LocalToolRuntime(),
            "remote": HTTPRemoteToolRuntime(),
            "mcp": MCPToolRuntime(),
        }
        self._runtimes.update(runtimes or {})

    def register(self, definition: AnyToolDefinition) -> AnyToolDefinition:
        self._tools[definition.name] = definition
        return definition

    def get(self, name: str) -> AnyToolDefinition | None:
        return self._tools.get(name)

    def items(self) -> list[tuple[str, AnyToolDefinition]]:
        return list(self._tools.items())

    async def __aenter__(self) -> "ToolRegistry":
        return self

    async def __aexit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        await self.aclose()

    def merge(self, tools: ToolSet | "ToolRegistry" | None) -> "ToolRegistry":
        merged = ToolRegistry(self._tools, runtimes=self._runtimes)
        if isinstance(tools, ToolRegistry):
            for definition in tools._tools.values():
                merged.register(definition)
            merged._runtimes.update(tools._runtimes)
            return merged
        for definition in dict(tools or {}).values():
            merged.register(definition)
        return merged

    async def execute(self, definition: AnyToolDefinition, input: Any, context: ToolExecutionContext) -> Any:
        if not is_callable_tool_definition(definition):
            raise RuntimeError(f'Tool "{definition.name}" is provider-managed and cannot run in the local agent runtime.')
        runtime = self._runtimes.get(definition.source, UnsupportedToolRuntime(definition.source))
        return await runtime.execute(definition, input, context)

    async def aclose(self) -> None:
        seen: set[int] = set()
        for runtime in self._runtimes.values():
            marker = id(runtime)
            if marker in seen:
                continue
            seen.add(marker)
            await runtime.aclose()


def _validate_postgres_table_prefix(table_prefix: str) -> str:
    if not _POSTGRES_IDENTIFIER_RE.match(table_prefix):
        raise ValidationError(
            'The "table_prefix" field must match the SQL identifier pattern [A-Za-z_][A-Za-z0-9_]*.'
        )
    return table_prefix


def mcp_stdio_server(
    *,
    name: str,
    command: str,
    args: Iterable[str] | None = None,
    env: dict[str, str] | None = None,
    timeout_ms: int | None = None,
) -> MCPServerConfig:
    return MCPServerConfig(
        transport="stdio",
        name=name,
        command=command,
        args=list(args or []),
        env=dict(env or {}),
        timeout_ms=timeout_ms,
    )


def mcp_http_server(
    *,
    name: str,
    url: str,
    headers: dict[str, str] | None = None,
    timeout_ms: int | None = None,
) -> MCPServerConfig:
    return MCPServerConfig(
        transport="streamable-http",
        name=name,
        url=url,
        headers=dict(headers or {}),
        timeout_ms=timeout_ms,
    )


def _sanitize_tool_name(value: str) -> str:
    normalized: list[str] = []
    last_was_separator = False
    for char in value:
        if char.isalnum():
            normalized.append(char.lower())
            last_was_separator = False
            continue
        if not last_was_separator:
            normalized.append("_")
            last_was_separator = True
    sanitized = "".join(normalized).strip("_")
    return sanitized or "tool"


def _build_mcp_local_tool_name(
    tool_name: str,
    *,
    prefix: str | None,
    name_transform: Literal["preserve", "snake_case"],
) -> str:
    base_name = _sanitize_tool_name(tool_name) if name_transform == "snake_case" else str(tool_name)
    resolved_prefix = _sanitize_tool_name(prefix) if prefix and name_transform == "snake_case" else prefix
    if not resolved_prefix:
        return base_name
    if resolved_prefix.endswith("_"):
        return f"{resolved_prefix}{base_name}"
    return f"{resolved_prefix}_{base_name}"


async def _load_mcp_tool_definitions(
    server: MCPServerConfig,
    *,
    prefix: str | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    name_transform: Literal["preserve", "snake_case"] = "preserve",
) -> ToolSet:
    runtime = MCPToolRuntime()
    try:
        include_set = set(include or [])
        exclude_set = set(exclude or [])
        tools: ToolSet = {}
        seen_remote_names: dict[str, str] = {}
        for item in await runtime.list_tools(server):
            tool_name = getattr(item, "name", None)
            if tool_name is None and isinstance(item, dict):
                tool_name = item.get("name")
            if not tool_name:
                continue
            remote_name = str(tool_name)
            if include_set and remote_name not in include_set:
                continue
            if remote_name in exclude_set:
                continue
            description = getattr(item, "description", None)
            if description is None and isinstance(item, dict):
                description = item.get("description")
            schema = getattr(item, "inputSchema", None)
            if schema is None:
                schema = getattr(item, "input_schema", None)
            if schema is None and isinstance(item, dict):
                schema = item.get("inputSchema") or item.get("input_schema") or {}
            local_name = _build_mcp_local_tool_name(remote_name, prefix=prefix, name_transform=name_transform)
            previous_remote_name = seen_remote_names.get(local_name)
            if previous_remote_name is not None and previous_remote_name != remote_name:
                raise ValidationError(
                    f'MCP tool name collision for "{local_name}": "{previous_remote_name}" and "{remote_name}". '
                    "Use a prefix or preserve the original names to disambiguate them."
                )
            seen_remote_names[local_name] = remote_name
            tools[local_name] = ToolDefinition(
                name=local_name,
                description=description,
                schema=schema or {},
                execute=None,
                source="mcp",
                metadata={"mcp_server": server.name, "mcp_tool_name": remote_name},
                mcp_config=MCPToolConfig(server=server, tool_name=remote_name),
            )
        return tools
    finally:
        await runtime.aclose()


async def discover_mcp_tools(
    server: MCPServerConfig,
    *,
    prefix: str | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> ToolSet:
    return await _load_mcp_tool_definitions(
        server,
        prefix=prefix,
        include=include,
        exclude=exclude,
        name_transform="preserve",
    )


async def create_mcp_tool_registry(
    server: MCPServerConfig,
    *,
    prefix: str | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    name_transform: Literal["preserve", "snake_case"] = "snake_case",
) -> ToolRegistry:
    resolved_prefix = server.name if prefix is None and name_transform == "snake_case" else prefix
    tools = await _load_mcp_tool_definitions(
        server,
        prefix=resolved_prefix,
        include=include,
        exclude=exclude,
        name_transform=name_transform,
    )
    return ToolRegistry(
        tools,
        runtimes={"mcp": MCPToolRuntime()},
    )


@dataclass(slots=True)
class Agent:
    name: str
    model: LanguageModel | RealtimeModel
    instructions: str | None = None
    tools: ToolSet | ToolRegistry = field(default_factory=dict)
    skills: SkillSet | SkillRegistry = field(default_factory=dict)
    subagents: dict[str, "Agent"] = field(default_factory=dict)
    memory: AgentMemory | None = None
    checkpoint_store: AgentCheckpointStore | None = None
    run_store: AgentRunStore | None = None
    approval_policy: ApprovalPolicy | None = None
    input_guardrails: list[InputGuardrail] = field(default_factory=list)
    output_guardrails: list[OutputGuardrail] = field(default_factory=list)
    run_limits: RunLimits = field(default_factory=RunLimits)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentCheckpoint:
    run_id: str
    session_id: str
    agent_name: str
    step_index: int
    request: ModelGenerateInput
    response: Any
    saved_at_ms: int
    is_final: bool = False


@dataclass(slots=True)
class AgentRunStartEvent:
    type: str = "run-start"
    run_id: str = ""
    session_id: str = ""
    agent_name: str = ""


@dataclass(slots=True)
class AgentDelegationStartEvent:
    type: str = "delegation-start"
    agent_name: str = ""
    handoff_depth: int = 0


@dataclass(slots=True)
class AgentDelegationFinishEvent:
    type: str = "delegation-finish"
    agent_name: str = ""
    handoff_depth: int = 0
    finish_reason: FinishReason | None = None


@dataclass(slots=True)
class AgentTextDeltaEvent:
    type: str = "text-delta"
    text_delta: str = ""


@dataclass(slots=True)
class AgentToolCallEvent:
    type: str = "tool-call"
    tool_call: ToolCall = field(default_factory=lambda: ToolCall(id="", name="", input={}))


@dataclass(slots=True)
class AgentToolApprovalEvent:
    type: str = "tool-approval"
    tool_name: str = ""
    tool_input: Any = None
    approved: bool = True
    reason: str | None = None
    provider: str | None = None
    provider_managed: bool = False
    approval_request_id: str | None = None
    tool_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentToolResultEvent:
    type: str = "tool-result"
    tool_result: ToolExecutionResult = field(
        default_factory=lambda: ToolExecutionResult(tool_call_id="", tool_name="", is_error=False)
    )


@dataclass(slots=True)
class AgentSkillActivatedEvent:
    type: str = "skill-activated"
    skill_name: str = ""
    activation: SkillActivationMode = "explicit"
    path: str | None = None
    description: str | None = None


@dataclass(slots=True)
class AgentSkillResolvedEvent:
    type: str = "skill-resolved"
    skill_name: str = ""
    skill_version: str | None = None
    entrypoints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentSkillDependencyCheckEvent:
    type: str = "skill-dependency-check"
    skill_name: str = ""
    dependency_type: str = ""
    dependency_value: str = ""
    available: bool = True


@dataclass(slots=True)
class AgentSkillSkippedEvent:
    type: str = "skill-skipped"
    skill_name: str = ""
    activation: SkillActivationMode = "sticky"
    reason: str = ""
    path: str | None = None


@dataclass(slots=True)
class AgentSkillExecutionStartEvent:
    type: str = "skill-execution-start"
    skill_name: str = ""
    entrypoint: str = ""


@dataclass(slots=True)
class AgentSkillExecutionFinishEvent:
    type: str = "skill-execution-finish"
    skill_name: str = ""
    entrypoint: str = ""
    ok: bool = True


@dataclass(slots=True)
class AgentSkillArtifactCreatedEvent:
    type: str = "skill-artifact-created"
    skill_name: str = ""
    entrypoint: str = ""
    artifact: SkillArtifact | None = None


@dataclass(slots=True)
class AgentGuardrailEvent:
    type: str = "guardrail"
    stage: Literal["input", "output"] = "input"
    guardrail_name: str = ""
    triggered: bool = False
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentCheckpointEvent:
    type: str = "checkpoint"
    checkpoint: AgentCheckpoint | None = None


@dataclass(slots=True)
class AgentSummaryUpdateEvent:
    type: str = "summary-update"
    summary: str | None = None


@dataclass(slots=True)
class AgentHandoffRequestedEvent:
    type: str = "handoff-requested"
    handoff: AgentHandoff | None = None


@dataclass(slots=True)
class AgentHandoffResolvedEvent:
    type: str = "handoff-resolved"
    source_agent: str = ""
    target_agent: str = ""


@dataclass(slots=True)
class AgentHandoffFailedEvent:
    type: str = "handoff-failed"
    source_agent: str = ""
    target_agent: str = ""
    reason: str | None = None


@dataclass(slots=True)
class AgentHandoffEvent:
    type: str = "handoff"
    handoff: AgentHandoff | None = None


@dataclass(slots=True)
class AgentFinishEvent:
    type: str = "finish"
    run_id: str = ""
    session_id: str = ""
    text: str = ""
    finish_reason: FinishReason | None = None


@dataclass(slots=True)
class AgentErrorEvent:
    type: str = "error"
    error: Exception | None = None


AgentEvent: TypeAlias = (
    AgentRunStartEvent
    | AgentDelegationStartEvent
    | AgentDelegationFinishEvent
    | AgentTextDeltaEvent
    | AgentToolCallEvent
    | AgentToolApprovalEvent
    | AgentToolResultEvent
    | AgentSkillActivatedEvent
    | AgentSkillResolvedEvent
    | AgentSkillDependencyCheckEvent
    | AgentSkillSkippedEvent
    | AgentSkillExecutionStartEvent
    | AgentSkillExecutionFinishEvent
    | AgentSkillArtifactCreatedEvent
    | AgentGuardrailEvent
    | AgentCheckpointEvent
    | AgentSummaryUpdateEvent
    | AgentHandoffRequestedEvent
    | AgentHandoffResolvedEvent
    | AgentHandoffFailedEvent
    | AgentHandoffEvent
    | AgentFinishEvent
    | AgentErrorEvent
)


@dataclass(slots=True)
class AgentTraceSegment:
    agent_name: str
    started_at_ms: int
    finished_at_ms: int | None = None


@dataclass(slots=True)
class AgentTrace:
    run_id: str
    session_id: str
    agent_name: str
    started_at_ms: int
    finished_at_ms: int | None = None
    events: list[AgentEvent] = field(default_factory=list)
    orchestration_path: list[str] = field(default_factory=list)
    segments: list[AgentTraceSegment] = field(default_factory=list)
    tool_call_count: int = 0
    approval_count: int = 0
    guardrail_trigger_count: int = 0
    checkpoint_count: int = 0
    handoff_count: int = 0


@dataclass(slots=True)
class AgentRunResult:
    run_id: str
    agent_name: str
    session: AgentSession
    text: str
    finish_reason: FinishReason | None = None
    provider_finish_reason: str | None = None
    usage: TokenUsage | None = None
    steps: list[GenerateTextStep] = field(default_factory=list)
    messages: list[ModelMessage] = field(default_factory=list)
    tool_results: list[ToolExecutionResult] = field(default_factory=list)
    artifacts: list[SkillArtifact] = field(default_factory=list)
    trace: AgentTrace | None = None
    handoff: AgentHandoff | None = None
    orchestration_path: list[str] = field(default_factory=list)
    resumed_from_checkpoint: AgentCheckpoint | None = None
    state: AgentRunState | None = None


class InMemoryAgentMemory:
    def __init__(self, *, summary_config: SummaryConfig | None = None) -> None:
        self.summary_config = summary_config or SummaryConfig()
        self._store: dict[str, AgentMemoryState] = {}

    async def load(self, session_id: str) -> AgentMemoryState:
        state = self._store.get(session_id)
        if state is None:
            return AgentMemoryState()
        return AgentMemoryState(messages=list(state.messages), summary=state.summary, metadata=dict(state.metadata))

    async def save(self, session_id: str, state: AgentMemoryState) -> None:
        self._store[session_id] = AgentMemoryState(
            messages=list(state.messages),
            summary=state.summary,
            metadata=dict(state.metadata),
        )

    async def summarize(
        self,
        *,
        session_id: str,
        state: AgentMemoryState,
        agent: Agent,
    ) -> str | None:
        if not state.messages:
            return state.summary
        older_messages = state.messages[:-self.summary_config.preserve_recent_messages]
        if not older_messages:
            return state.summary
        existing = state.summary.strip() if state.summary else ""
        transcript = _message_text(older_messages)
        parts = [part for part in [existing, transcript] if part]
        if not parts:
            return state.summary
        summary = "\n".join(parts)
        if len(summary) > self.summary_config.max_summary_chars:
            summary = summary[: self.summary_config.max_summary_chars]
        return summary.strip() or None


class InMemoryAgentCheckpointStore:
    def __init__(self) -> None:
        self._items: list[AgentCheckpoint] = []

    async def save(self, checkpoint: AgentCheckpoint) -> None:
        self._items.append(checkpoint)

    async def get_latest(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentCheckpoint | None:
        items = await self.list(session_id=session_id, run_id=run_id)
        if not items:
            return None
        return max(items, key=lambda item: (item.saved_at_ms, item.step_index))

    async def list(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> list[AgentCheckpoint]:
        items = list(self._items)
        if session_id is not None:
            items = [item for item in items if item.session_id == session_id]
        if run_id is not None:
            items = [item for item in items if item.run_id == run_id]
        return items


def create_agent_session(
    *,
    id: str | None = None,
    messages: list[ModelMessage] | None = None,
    summary: str | None = None,
    state: dict[str, JsonValue] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentSession:
    return AgentSession(
        id=id or _new_id("session"),
        messages=list(messages or []),
        summary=summary,
        state=dict(state or {}),
        metadata=dict(metadata or {}),
    )


def set_agent_session_skills(session: AgentSession, *skill_names: str) -> AgentSession:
    names = _normalize_skill_names(skill_names)
    session.metadata = {
        **session.metadata,
        "sticky_skills": names,
        "active_skills": [],
    }
    return session


def get_agent_session_skills(session: AgentSession) -> list[str]:
    return _sticky_skill_names(session)


def clear_agent_session_skills(session: AgentSession) -> AgentSession:
    session.metadata = {
        **session.metadata,
        "sticky_skills": [],
        "active_skills": [],
    }
    return session


def create_in_memory_agent_memory_store(*, summary_config: SummaryConfig | None = None) -> InMemoryAgentMemory:
    return InMemoryAgentMemory(summary_config=summary_config)


def create_in_memory_checkpoint_store() -> InMemoryAgentCheckpointStore:
    return InMemoryAgentCheckpointStore()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _json_loads(value: str | None) -> Any:
    if not value:
        return None
    return json.loads(value)


def _coerce_json_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        decoded = _json_loads(value)
        return dict(decoded or {})
    return dict(value or {})


def _serialize_agent_memory_state(state: AgentMemoryState) -> dict[str, Any]:
    return {
        "messages": serialize_messages(state.messages),
        "summary": state.summary,
        "metadata": dict(state.metadata),
    }


def _deserialize_agent_memory_state(payload: dict[str, Any] | None) -> AgentMemoryState:
    if payload is None:
        return AgentMemoryState()
    return AgentMemoryState(
        messages=deserialize_messages(payload.get("messages")),
        summary=payload.get("summary"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _serialize_agent_checkpoint(checkpoint: AgentCheckpoint) -> dict[str, Any]:
    return {
        "run_id": checkpoint.run_id,
        "session_id": checkpoint.session_id,
        "agent_name": checkpoint.agent_name,
        "step_index": checkpoint.step_index,
        "request": serialize_model_generate_input(checkpoint.request),
        "response": serialize_generate_result(checkpoint.response),
        "saved_at_ms": checkpoint.saved_at_ms,
        "is_final": checkpoint.is_final,
    }


def _deserialize_agent_checkpoint(payload: dict[str, Any]) -> AgentCheckpoint:
    return AgentCheckpoint(
        run_id=str(payload.get("run_id", "")),
        session_id=str(payload.get("session_id", "")),
        agent_name=str(payload.get("agent_name", "")),
        step_index=int(payload.get("step_index", 0)),
        request=deserialize_model_generate_input(dict(payload.get("request") or {})),
        response=deserialize_generate_result(dict(payload.get("response") or {})),
        saved_at_ms=int(payload.get("saved_at_ms", 0)),
        is_final=bool(payload.get("is_final", False)),
    )


class SQLiteAgentMemoryStore:
    def __init__(self, path: str, *, summary_config: SummaryConfig | None = None, namespace: str = "default") -> None:
        self.summary_config = summary_config or SummaryConfig()
        self._path = path
        self._namespace = namespace
        self._ready = False

    async def _execute(self, sql: str, params: tuple[Any, ...] = (), *, fetchone: bool = False) -> Any:
        def runner() -> Any:
            connection = sqlite3.connect(self._path)
            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS zhivex_agent_memory (
                        namespace TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        state_json TEXT NOT NULL,
                        updated_at_ms INTEGER NOT NULL,
                        PRIMARY KEY (namespace, session_id)
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS zhivex_agent_memory_updated_idx ON zhivex_agent_memory (updated_at_ms)"
                )
                cursor = connection.execute(sql, params)
                row = cursor.fetchone() if fetchone else None
                connection.commit()
                return row
            finally:
                connection.close()

        return await asyncio.to_thread(runner)

    async def load(self, session_id: str) -> AgentMemoryState:
        row = await self._execute(
            "SELECT state_json FROM zhivex_agent_memory WHERE namespace = ? AND session_id = ?",
            (self._namespace, session_id),
            fetchone=True,
        )
        if row is None:
            return AgentMemoryState()
        return _deserialize_agent_memory_state(_json_loads(row[0]))

    async def save(self, session_id: str, state: AgentMemoryState) -> None:
        payload = _json_dumps(_serialize_agent_memory_state(state))
        await self._execute(
            """
            INSERT INTO zhivex_agent_memory (namespace, session_id, state_json, updated_at_ms)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(namespace, session_id)
            DO UPDATE SET state_json = excluded.state_json, updated_at_ms = excluded.updated_at_ms
            """,
            (self._namespace, session_id, payload, _now_ms()),
        )

    async def summarize(
        self,
        *,
        session_id: str,
        state: AgentMemoryState,
        agent: "Agent",
    ) -> str | None:
        return await InMemoryAgentMemory(summary_config=self.summary_config).summarize(
            session_id=session_id,
            state=state,
            agent=agent,
        )


class SQLiteAgentCheckpointStore:
    def __init__(self, path: str, *, namespace: str = "default") -> None:
        self._path = path
        self._namespace = namespace

    async def _execute(self, sql: str, params: tuple[Any, ...] = (), *, fetchone: bool = False, fetchall: bool = False) -> Any:
        def runner() -> Any:
            connection = sqlite3.connect(self._path)
            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS zhivex_agent_checkpoints (
                        namespace TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        agent_name TEXT NOT NULL,
                        step_index INTEGER NOT NULL,
                        saved_at_ms INTEGER NOT NULL,
                        is_final INTEGER NOT NULL,
                        checkpoint_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS zhivex_agent_checkpoints_session_idx
                    ON zhivex_agent_checkpoints (namespace, session_id, saved_at_ms, step_index)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS zhivex_agent_checkpoints_run_idx
                    ON zhivex_agent_checkpoints (namespace, run_id, saved_at_ms, step_index)
                    """
                )
                cursor = connection.execute(sql, params)
                if fetchone:
                    row = cursor.fetchone()
                elif fetchall:
                    row = cursor.fetchall()
                else:
                    row = None
                connection.commit()
                return row
            finally:
                connection.close()

        return await asyncio.to_thread(runner)

    async def save(self, checkpoint: AgentCheckpoint) -> None:
        payload = _json_dumps(_serialize_agent_checkpoint(checkpoint))
        await self._execute(
            """
            INSERT INTO zhivex_agent_checkpoints (
                namespace, run_id, session_id, agent_name, step_index, saved_at_ms, is_final, checkpoint_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._namespace,
                checkpoint.run_id,
                checkpoint.session_id,
                checkpoint.agent_name,
                checkpoint.step_index,
                checkpoint.saved_at_ms,
                1 if checkpoint.is_final else 0,
                payload,
            ),
        )

    async def get_latest(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentCheckpoint | None:
        if session_id is None and run_id is None:
            raise ValidationError('Pass either "session_id" or "run_id" to get_latest().')
        if session_id is not None:
            row = await self._execute(
                """
                SELECT checkpoint_json
                FROM zhivex_agent_checkpoints
                WHERE namespace = ? AND session_id = ?
                ORDER BY saved_at_ms DESC, step_index DESC
                LIMIT 1
                """,
                (self._namespace, session_id),
                fetchone=True,
            )
        else:
            row = await self._execute(
                """
                SELECT checkpoint_json
                FROM zhivex_agent_checkpoints
                WHERE namespace = ? AND run_id = ?
                ORDER BY saved_at_ms DESC, step_index DESC
                LIMIT 1
                """,
                (self._namespace, run_id),
                fetchone=True,
            )
        if row is None:
            return None
        return _deserialize_agent_checkpoint(_json_loads(row[0]))

    async def list(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> list[AgentCheckpoint]:
        sql = "SELECT checkpoint_json FROM zhivex_agent_checkpoints WHERE namespace = ?"
        params: list[Any] = [self._namespace]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " ORDER BY saved_at_ms ASC, step_index ASC"
        rows = await self._execute(sql, tuple(params), fetchall=True)
        return [_deserialize_agent_checkpoint(_json_loads(row[0])) for row in rows or []]


class PostgresAgentMemoryStore:
    def __init__(self, dsn: str, *, summary_config: SummaryConfig | None = None, table_prefix: str = "zhivex_ai") -> None:
        self.summary_config = summary_config or SummaryConfig()
        self._dsn = dsn
        self._table_prefix = _validate_postgres_table_prefix(table_prefix)

    def _table(self) -> str:
        return f"{self._table_prefix}_agent_memory"

    async def _connect(self) -> Any:
        try:
            import asyncpg  # type: ignore[import-not-found]
        except Exception as error:
            raise RuntimeError('Postgres support requires the optional dependency "asyncpg".') from error
        return await asyncpg.connect(self._dsn)

    async def _ensure_schema(self, connection: Any) -> None:
        table = self._table()
        await connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                session_id TEXT PRIMARY KEY,
                state_json JSONB NOT NULL,
                updated_at_ms BIGINT NOT NULL
            )
            """
        )
        await connection.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_updated_idx ON {table} (updated_at_ms)"
        )

    async def load(self, session_id: str) -> AgentMemoryState:
        connection = await self._connect()
        try:
            await self._ensure_schema(connection)
            row = await connection.fetchrow(
                f"SELECT state_json FROM {self._table()} WHERE session_id = $1",
                session_id,
            )
        finally:
            await connection.close()
        if row is None:
            return AgentMemoryState()
        return _deserialize_agent_memory_state(_coerce_json_payload(row["state_json"]))

    async def save(self, session_id: str, state: AgentMemoryState) -> None:
        connection = await self._connect()
        try:
            await self._ensure_schema(connection)
            await connection.execute(
                f"""
                INSERT INTO {self._table()} (session_id, state_json, updated_at_ms)
                VALUES ($1, $2::jsonb, $3)
                ON CONFLICT(session_id)
                DO UPDATE SET state_json = EXCLUDED.state_json, updated_at_ms = EXCLUDED.updated_at_ms
                """,
                session_id,
                _json_dumps(_serialize_agent_memory_state(state)),
                _now_ms(),
            )
        finally:
            await connection.close()

    async def summarize(
        self,
        *,
        session_id: str,
        state: AgentMemoryState,
        agent: "Agent",
    ) -> str | None:
        return await InMemoryAgentMemory(summary_config=self.summary_config).summarize(
            session_id=session_id,
            state=state,
            agent=agent,
        )


class PostgresAgentCheckpointStore:
    def __init__(self, dsn: str, *, table_prefix: str = "zhivex_ai") -> None:
        self._dsn = dsn
        self._table_prefix = _validate_postgres_table_prefix(table_prefix)

    def _table(self) -> str:
        return f"{self._table_prefix}_agent_checkpoints"

    async def _connect(self) -> Any:
        try:
            import asyncpg
        except Exception as error:
            raise RuntimeError('Postgres support requires the optional dependency "asyncpg".') from error
        return await asyncpg.connect(self._dsn)

    async def _ensure_schema(self, connection: Any) -> None:
        table = self._table()
        await connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                run_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                saved_at_ms BIGINT NOT NULL,
                is_final BOOLEAN NOT NULL,
                checkpoint_json JSONB NOT NULL
            )
            """
        )
        await connection.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_session_idx ON {table} (session_id, saved_at_ms, step_index)"
        )
        await connection.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_run_idx ON {table} (run_id, saved_at_ms, step_index)"
        )

    async def save(self, checkpoint: AgentCheckpoint) -> None:
        connection = await self._connect()
        try:
            await self._ensure_schema(connection)
            await connection.execute(
                f"""
                INSERT INTO {self._table()} (
                    run_id, session_id, agent_name, step_index, saved_at_ms, is_final, checkpoint_json
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                """,
                checkpoint.run_id,
                checkpoint.session_id,
                checkpoint.agent_name,
                checkpoint.step_index,
                checkpoint.saved_at_ms,
                checkpoint.is_final,
                _json_dumps(_serialize_agent_checkpoint(checkpoint)),
            )
        finally:
            await connection.close()

    async def get_latest(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentCheckpoint | None:
        if session_id is None and run_id is None:
            raise ValidationError('Pass either "session_id" or "run_id" to get_latest().')
        connection = await self._connect()
        try:
            await self._ensure_schema(connection)
            if session_id is not None:
                row = await connection.fetchrow(
                    f"""
                    SELECT checkpoint_json
                    FROM {self._table()}
                    WHERE session_id = $1
                    ORDER BY saved_at_ms DESC, step_index DESC
                    LIMIT 1
                    """,
                    session_id,
                )
            else:
                row = await connection.fetchrow(
                    f"""
                    SELECT checkpoint_json
                    FROM {self._table()}
                    WHERE run_id = $1
                    ORDER BY saved_at_ms DESC, step_index DESC
                    LIMIT 1
                    """,
                    run_id,
                )
        finally:
            await connection.close()
        if row is None:
            return None
        return _deserialize_agent_checkpoint(_coerce_json_payload(row["checkpoint_json"]))

    async def list(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> list[AgentCheckpoint]:
        connection = await self._connect()
        try:
            await self._ensure_schema(connection)
            if session_id is not None and run_id is not None:
                rows = await connection.fetch(
                    f"""
                    SELECT checkpoint_json
                    FROM {self._table()}
                    WHERE session_id = $1 AND run_id = $2
                    ORDER BY saved_at_ms ASC, step_index ASC
                    """,
                    session_id,
                    run_id,
                )
            elif session_id is not None:
                rows = await connection.fetch(
                    f"""
                    SELECT checkpoint_json
                    FROM {self._table()}
                    WHERE session_id = $1
                    ORDER BY saved_at_ms ASC, step_index ASC
                    """,
                    session_id,
                )
            elif run_id is not None:
                rows = await connection.fetch(
                    f"""
                    SELECT checkpoint_json
                    FROM {self._table()}
                    WHERE run_id = $1
                    ORDER BY saved_at_ms ASC, step_index ASC
                    """,
                    run_id,
                )
            else:
                rows = await connection.fetch(
                    f"""
                    SELECT checkpoint_json
                    FROM {self._table()}
                    ORDER BY saved_at_ms ASC, step_index ASC
                    """
                )
        finally:
            await connection.close()
        return [_deserialize_agent_checkpoint(_coerce_json_payload(row["checkpoint_json"])) for row in rows]


def create_sqlite_agent_memory_store(
    path: str,
    *,
    summary_config: SummaryConfig | None = None,
    namespace: str = "default",
) -> SQLiteAgentMemoryStore:
    return SQLiteAgentMemoryStore(path, summary_config=summary_config, namespace=namespace)


def create_sqlite_checkpoint_store(path: str, *, namespace: str = "default") -> SQLiteAgentCheckpointStore:
    return SQLiteAgentCheckpointStore(path, namespace=namespace)


def create_postgres_agent_memory_store(
    dsn: str,
    *,
    summary_config: SummaryConfig | None = None,
    table_prefix: str = "zhivex_ai",
) -> PostgresAgentMemoryStore:
    return PostgresAgentMemoryStore(dsn, summary_config=summary_config, table_prefix=table_prefix)


def create_postgres_checkpoint_store(
    dsn: str,
    *,
    table_prefix: str = "zhivex_ai",
) -> PostgresAgentCheckpointStore:
    return PostgresAgentCheckpointStore(dsn, table_prefix=table_prefix)


async def allow_all_approval_policy(request: ToolApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision(approved=True)


async def deny_all_approval_policy(request: ToolApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision(approved=False, reason="Tool execution denied by policy.")


def permission_allowlist_approval_policy(*allowed_permissions: str) -> ApprovalPolicy:
    allowed = set(allowed_permissions)

    async def policy(request: ToolApprovalRequest) -> ApprovalDecision:
        if not request.tool_permissions:
            return ApprovalDecision(approved=True)
        missing = [permission for permission in request.tool_permissions if permission not in allowed]
        if missing:
            return ApprovalDecision(approved=False, reason=f"Missing permissions: {', '.join(missing)}")
        return ApprovalDecision(approved=True)

    return policy


async def load_agent_session(agent: Agent, session_id: str, *, metadata: dict[str, Any] | None = None) -> AgentSession:
    messages: list[ModelMessage] = []
    summary: str | None = None
    workflow_state: dict[str, JsonValue] = {}
    merged_metadata = dict(metadata or {})
    if agent.memory is not None:
        state = await agent.memory.load(session_id)
        messages = list(state.messages)
        summary = state.summary
        raw_workflow_state = state.metadata.get("state") or state.metadata.get("workflow_state")
        if isinstance(raw_workflow_state, dict):
            workflow_state = dict(raw_workflow_state)
        merged_metadata = {**state.metadata, **merged_metadata}
    return AgentSession(id=session_id, messages=messages, summary=summary, state=workflow_state, metadata=merged_metadata)


def _normalize_approval_decision(value: ApprovalDecision | bool | None) -> ApprovalDecision:
    if isinstance(value, ApprovalDecision):
        return value
    if value is False:
        return ApprovalDecision(approved=False)
    return ApprovalDecision(approved=True)


def _effective_max_steps(limits: RunLimits, requested: int | None) -> int | None:
    if limits.max_steps is None:
        return requested
    if requested is None:
        return limits.max_steps
    return min(limits.max_steps, requested)


def _effective_timeout_ms(limits: RunLimits, requested: int | None) -> int | None:
    if limits.max_wall_time_ms is None:
        return requested
    if requested is None:
        return limits.max_wall_time_ms
    return min(limits.max_wall_time_ms, requested)


def _detect_handoff(tool_results: list[ToolExecutionResult]) -> AgentHandoff | None:
    for result in reversed(tool_results):
        if isinstance(result.output, dict) and result.output.get(HANDOFF_MARKER):
            handoff_input = result.output.get("input")
            handoff_metadata = result.output.get("metadata")
            return AgentHandoff(
                target_agent=str(result.output.get("target_agent")),
                input=handoff_input if isinstance(handoff_input, str) else None,
                metadata=dict(handoff_metadata) if isinstance(handoff_metadata, dict) else {},
            )
    return None


def _int_from_json(value: JsonValue | None, default: int = 0) -> int:
    if isinstance(value, (str, int, float)):
        return int(value)
    return default


def _should_refresh_summary(memory: AgentMemory, session: AgentSession) -> bool:
    config = memory.summary_config
    if len(session.messages) > config.max_messages:
        return True
    text_length = sum(len(_text_from_message(message)) for message in session.messages)
    return text_length > config.max_summary_chars * 2


def _context_messages(session: AgentSession, memory: AgentMemory | None) -> list[ModelMessage]:
    if memory is None:
        return list(session.messages)
    preserve = max(0, memory.summary_config.preserve_recent_messages)
    if preserve == 0:
        return []
    return list(session.messages[-preserve:])


def _build_run_messages(
    *,
    agent: Agent,
    session: AgentSession,
    prompt: str | None,
    messages: list[ModelMessage] | None,
    active_skills: list[SkillDefinition] | None = None,
) -> list[ModelMessage]:
    if prompt is not None and messages is not None:
        raise ValidationError('Pass either "prompt" or "messages", but not both.')

    built: list[ModelMessage] = []
    if agent.instructions:
        built.append(create_text_message("system", agent.instructions))
    for active_skill in active_skills or []:
        built.append(create_text_message("system", _skill_system_message(active_skill)))
    if session.summary:
        built.append(create_text_message("system", f"{SUMMARY_MARKER}{session.summary}"))
    built.extend(_context_messages(session, agent.memory))
    if messages is not None:
        built.extend(messages)
    elif prompt is not None:
        built.append(create_text_message("user", prompt))
    return built


def _guardrail_name(guardrail: Any) -> str:
    if hasattr(guardrail, "__name__"):
        return str(getattr(guardrail, "__name__"))
    return guardrail.__class__.__name__


def _normalize_guardrail_result(value: GuardrailResult | bool | None) -> GuardrailResult:
    if isinstance(value, GuardrailResult):
        return value
    if isinstance(value, bool):
        return GuardrailResult(tripwire_triggered=value)
    if value is None:
        return GuardrailResult(tripwire_triggered=False)
    raise TypeError("Guardrails must return GuardrailResult, bool, or None.")


@dataclass(slots=True)
class _ProviderManagedApproval:
    provider: str
    approval_request_id: str
    tool_name: str
    tool_input: Any
    server_label: str | None
    raw_payload: Any


@dataclass(slots=True)
class _ProviderManagedToolTraceEvent:
    provider: str
    event_key: str
    tool_call: ToolCall
    raw_payload: Any


def _decode_provider_managed_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return arguments


def _parse_provider_managed_approval_part(part: Any) -> _ProviderManagedApproval | None:
    if getattr(part, "type", None) != "provider-data":
        return None
    provider = str(getattr(part, "provider", "") or "")
    parsed: Any
    if provider == "openai":
        from .providers.openai import parse_openai_provider_data_part

        parsed = parse_openai_provider_data_part(part)
    elif provider == "azure-openai":
        from .providers.azure_openai import parse_azure_openai_provider_data_part

        parsed = parse_azure_openai_provider_data_part(part)
    else:
        return None
    if parsed is None or getattr(parsed, "type", None) != "mcp_approval_request":
        return None
    return _ProviderManagedApproval(
        provider=provider,
        approval_request_id=str(getattr(parsed, "id", "") or ""),
        tool_name=str(getattr(parsed, "name", "") or ""),
        tool_input=_decode_provider_managed_arguments(getattr(parsed, "arguments", "")),
        server_label=str(getattr(parsed, "server_label", "") or "") or None,
        raw_payload=parsed,
    )


def _provider_managed_event_key(provider: str, payload: Any) -> str:
    event_type = str(getattr(payload, "type", "") or payload.__class__.__name__)
    identifier = getattr(payload, "id", None) or getattr(payload, "approval_request_id", None) or getattr(payload, "response_id", None)
    if identifier:
        return f"{provider}:{event_type}:{identifier}"
    try:
        stable_payload = json.dumps(
            asdict(cast("DataclassInstance", payload)) if is_dataclass(payload) and not isinstance(payload, type) else payload,
            sort_keys=True,
            default=str,
        )
    except TypeError:
        stable_payload = repr(payload)
    return f"{provider}:{event_type}:{stable_payload}"


def _parse_provider_managed_tool_trace_part(part: Any) -> _ProviderManagedToolTraceEvent | None:
    if getattr(part, "type", None) != "provider-data":
        return None
    provider = str(getattr(part, "provider", "") or "")
    parsed: Any
    tool_call: ToolCall | None
    if provider == "openai":
        from .providers.openai import openai_provider_data_tool_call, parse_openai_provider_data_part

        parsed = parse_openai_provider_data_part(part)
        tool_call = openai_provider_data_tool_call(parsed) if parsed is not None else None
    elif provider == "azure-openai":
        from .providers.azure_openai import azure_openai_provider_data_tool_call, parse_azure_openai_provider_data_part

        parsed = parse_azure_openai_provider_data_part(part)
        tool_call = azure_openai_provider_data_tool_call(parsed) if parsed is not None else None
    else:
        return None
    if parsed is None or tool_call is None:
        return None
    return _ProviderManagedToolTraceEvent(
        provider=provider,
        event_key=_provider_managed_event_key(provider, parsed),
        tool_call=tool_call,
        raw_payload=parsed,
    )


def _provider_managed_approval_response_message(
    approval: _ProviderManagedApproval,
    *,
    approved: bool,
    reason: str | None = None,
) -> ModelMessage:
    if approval.provider == "openai":
        from .providers.openai import openai_mcp_approval_response

        part = openai_mcp_approval_response(
            approval_request_id=approval.approval_request_id,
            approve=approved,
            reason=reason,
        )
    else:
        from .providers.azure_openai import azure_openai_mcp_approval_response

        part = azure_openai_mcp_approval_response(
            approval_request_id=approval.approval_request_id,
            approve=approved,
            reason=reason,
        )
    return ModelMessage(role="assistant", parts=[part])


def _assistant_messages_from_result(result: GenerateTextOutput) -> list[ModelMessage]:
    messages = list(result.messages)
    if messages:
        return messages
    if result.steps:
        collected: list[ModelMessage] = []
        for step in result.steps:
            collected.extend(_response_messages(step))
        if collected:
            return collected
    if result.text:
        return [create_text_message("assistant", result.text)]
    return []


def _resolve_tool_registry(agent: Agent, extra_tools: ToolSet | ToolRegistry | None) -> ToolRegistry:
    base = agent.tools if isinstance(agent.tools, ToolRegistry) else ToolRegistry(agent.tools)
    return base.merge(extra_tools)


def _resolve_skill_registry(agent: Agent, extra_skills: SkillSet | SkillRegistry | None) -> SkillRegistry:
    base = agent.skills if isinstance(agent.skills, SkillRegistry) else SkillRegistry(agent.skills)
    return base.merge(extra_skills)


def _sticky_skill_names(session: AgentSession) -> list[str]:
    raw = session.metadata.get("sticky_skills")
    if not isinstance(raw, list):
        return []
    return _normalize_skill_names(raw)


async def _resolve_skill_tools(skill: SkillDefinition) -> ToolSet:
    resolved: ToolSet = dict(skill.tools)
    if skill.entrypoints:
        from .skillpacks import build_skill_entrypoint_tools

        resolved.update(build_skill_entrypoint_tools(skill))
    for dependency in skill.dependencies:
        if dependency.type != "mcp":
            continue
        if dependency.transport == "stdio":
            if not dependency.command:
                raise ValidationError(f'Skill "{skill.name}" requires "command" for stdio MCP dependencies.')
            server = mcp_stdio_server(
                name=dependency.value,
                command=dependency.command,
                args=dependency.args,
                env=dependency.env,
                timeout_ms=dependency.timeout_ms,
            )
        else:
            if not dependency.url:
                raise ValidationError(f'Skill "{skill.name}" requires "url" for HTTP MCP dependencies.')
            server = mcp_http_server(
                name=dependency.value,
                url=dependency.url,
                headers=dependency.headers,
                timeout_ms=dependency.timeout_ms,
            )
        prefix = dependency.prefix or f"{skill.name}_{dependency.value}"
        discovered = await discover_mcp_tools(
            server,
            prefix=prefix,
            include=dependency.include or None,
            exclude=dependency.exclude or None,
        )
        for name, definition in discovered.items():
            if name in resolved and resolved[name] != definition:
                raise ValidationError(f'Tool name collision while activating skill "{skill.name}": "{name}".')
            resolved[name] = definition
    return resolved


async def _select_active_skills(
    registry: SkillRegistry,
    *,
    agent: Agent,
    session: AgentSession,
    prompt: str | None,
    messages: list[ModelMessage] | None,
) -> tuple[list[_SkillActivation], list[_SkillSkip], ToolSet]:
    text = prompt or ""
    if messages is not None:
        message_text = _message_text(messages)
        text = f"{text}\n{message_text}".strip()
    selected: dict[str, _SkillActivation] = {}
    skipped: list[_SkillSkip] = []
    sticky_names = _sticky_skill_names(session)

    if text and registry.items():
        for _, definition in registry.items():
            if _explicit_skill_requested(definition, text):
                selected[definition.name] = _SkillActivation(skill=definition, mode="explicit")
            elif _should_activate_skill_implicitly(definition, text):
                selected.setdefault(definition.name, _SkillActivation(skill=definition, mode="implicit"))

    for sticky_name in sticky_names:
        if sticky_name in selected:
            continue
        sticky_skill = registry.get(sticky_name)
        if sticky_skill is None:
            skipped.append(_SkillSkip(skill_name=sticky_name, reason="sticky skill is no longer registered", mode="sticky"))
            continue
        selected[sticky_name] = _SkillActivation(skill=sticky_skill, mode="sticky")

    if not selected:
        return [], skipped, {}
    ordered_candidates = sorted(selected.values(), key=_skill_activation_sort_key)
    activations: list[_SkillActivation] = []
    resolved_tools: ToolSet = {}
    for activation in ordered_candidates:
        allowed, reason = _skill_allowed_for_agent(activation.skill, agent)
        if not allowed:
            if activation.mode == "explicit":
                raise ValidationError(f'Explicit skill "{activation.skill.name}" could not be activated because {reason}.')
            skipped.append(
                _SkillSkip(
                    skill_name=activation.skill.name,
                    reason=reason or "skill is not allowed for this agent",
                    mode=activation.mode,
                    path=activation.skill.path,
                )
            )
            continue
        try:
            candidate_tools = await _resolve_skill_tools(activation.skill)
        except Exception as error:
            reason = f"dependency resolution failed: {error}"
            if activation.mode == "explicit" or activation.skill.dependency_failure_mode == "fail":
                raise ValidationError(f'Skill "{activation.skill.name}" could not be activated: {reason}') from error
            skipped.append(
                _SkillSkip(
                    skill_name=activation.skill.name,
                    reason=reason,
                    mode=activation.mode,
                    path=activation.skill.path,
                )
            )
            continue
        conflicts = [name for name, definition in candidate_tools.items() if name in resolved_tools and resolved_tools[name] != definition]
        if conflicts:
            reason = f'conflicting tools: {", ".join(sorted(conflicts))}'
            if activation.mode == "explicit":
                raise ValidationError(f'Explicit skill "{activation.skill.name}" could not be activated due to {reason}.')
            skipped.append(
                _SkillSkip(
                    skill_name=activation.skill.name,
                    reason=reason,
                    mode=activation.mode,
                    path=activation.skill.path,
                )
            )
            continue
        resolved_tools.update(candidate_tools)
        activations.append(activation)
    return activations, skipped, resolved_tools


def _persist_active_skills(session: AgentSession, active_skills: list[_SkillActivation]) -> None:
    if not active_skills and "sticky_skills" not in session.metadata and "active_skills" not in session.metadata:
        return
    sticky_skill_names = [item.skill.name for item in active_skills if item.skill.persist_to_session]
    session.metadata = {
        **session.metadata,
        "active_skills": [
            {
                "name": item.skill.name,
                "path": item.skill.path,
                "metadata_path": item.skill.metadata_path,
                "version": item.skill.version,
                "entrypoints": [entrypoint.name for entrypoint in item.skill.entrypoints],
                "activation": item.mode,
                "priority": item.skill.priority,
            }
            for item in active_skills
        ],
        "sticky_skills": sticky_skill_names,
        "skill_versions": {item.skill.name: item.skill.version for item in active_skills if item.skill.version},
        "skill_entrypoints": {
            item.skill.name: [entrypoint.name for entrypoint in item.skill.entrypoints]
            for item in active_skills
            if item.skill.entrypoints
        },
    }


async def _emit_skill_events(
    *,
    active_skills: list[_SkillActivation],
    skipped_skills: list[_SkillSkip],
    emit: Callable[[AgentEvent], Awaitable[None]],
) -> None:
    for skipped in skipped_skills:
        await emit(
            AgentSkillSkippedEvent(
                skill_name=skipped.skill_name,
                activation=skipped.mode,
                reason=skipped.reason,
                path=skipped.path,
            )
        )
    for activation in active_skills:
        await emit(
            AgentSkillActivatedEvent(
                skill_name=activation.skill.name,
                activation=activation.mode,
                path=activation.skill.path,
                description=activation.skill.description,
            )
        )
        if activation.skill.version or activation.skill.entrypoints:
            await emit(
                AgentSkillResolvedEvent(
                    skill_name=activation.skill.name,
                    skill_version=activation.skill.version,
                    entrypoints=[item.name for item in activation.skill.entrypoints],
                )
            )
        for dependency in activation.skill.dependencies:
            await emit(
                AgentSkillDependencyCheckEvent(
                    skill_name=activation.skill.name,
                    dependency_type=dependency.type,
                    dependency_value=dependency.value,
                    available=True,
                )
            )


def _extract_skill_artifacts(tool_results: list[ToolExecutionResult]) -> list[SkillArtifact]:
    artifacts: list[SkillArtifact] = []
    for result in tool_results:
        if not isinstance(result.output, dict):
            continue
        payload = result.output.get("artifacts")
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            artifact_path = str(item.get("path") or "").strip()
            if not artifact_path:
                continue
            artifact_metadata = item.get("metadata")
            artifacts.append(
                SkillArtifact(
                    name=str(item.get("name") or Path(artifact_path).name),
                    path=artifact_path,
                    media_type=str(item.get("media_type") or "") or None,
                    role=str(item.get("role") or "primary"),  # type: ignore[arg-type]
                    description=str(item.get("description") or "") or None,
                    metadata=cast(dict[str, Any], artifact_metadata) if isinstance(artifact_metadata, dict) else {},
                )
            )
    return artifacts


def _extract_tool_calls_from_steps(steps: list[GenerateTextStep]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for step in steps:
        response_messages = step.response.messages or ([step.response.message] if step.response.message else [])
        for message in response_messages:
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    calls.append(part.tool_call)
    return calls


def _extract_provider_tool_results_from_steps(steps: list[GenerateTextStep]) -> list[ToolExecutionResult]:
    results: list[ToolExecutionResult] = []
    for step in steps:
        response_messages = step.response.messages or ([step.response.message] if step.response.message else [])
        for message in response_messages:
            for part in message.parts:
                if isinstance(part, ToolResultPart):
                    results.append(part.tool_result)
    return results


def _step_status_from_result(result: GenerateTextOutput) -> AgentRunStatus:
    return "failed" if result.finish_reason == "error" else "completed"


def _child_runs_from_tool_results(results: list[ToolExecutionResult], parent_run_id: str) -> list[AgentChildRun]:
    children: list[AgentChildRun] = []
    for result in results:
        output = result.output
        if not isinstance(output, dict):
            continue
        raw_child = output.get("child_run")
        if not isinstance(raw_child, dict):
            continue
        children.append(
            AgentChildRun(
                run_id=str(raw_child.get("run_id", "")),
                agent_name=str(raw_child.get("agent_name", "")),
                parent_run_id=str(raw_child.get("parent_run_id") or parent_run_id),
                status=cast(AgentRunStatus, raw_child.get("status", "completed")),
                output_text=str(raw_child.get("output_text", "")),
                tool_name=result.tool_name,
                error=str(raw_child.get("error")) if raw_child.get("error") is not None else None,
                steps=_int_from_json(raw_child.get("steps")),
                tool_calls=_int_from_json(raw_child.get("tool_calls")),
                tool_errors=_int_from_json(raw_child.get("tool_errors")),
            )
        )
    return children


def _agent_run_state_from_result(
    *,
    result: AgentRunResult,
    agent: Agent,
    parent_run_id: str | None,
    idempotency_key: str | None,
    started_at_ms: int,
    finished_at_ms: int,
    error: str | None = None,
) -> AgentRunState:
    steps = [
        AgentRunStep(
            index=index,
            status=_step_status_from_result(GenerateTextOutput(text=step.response.text or "", finish_reason=step.response.finish_reason)),
            tool_calls=_extract_tool_calls_from_steps([step]),
            tool_results=[],
            usage=step.response.usage,
            messages=_response_messages(step),
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
        )
        for index, step in enumerate(result.steps, start=1)
    ]
    status: AgentRunStatus = "failed" if error or result.finish_reason == "error" else "completed"
    state = AgentRunState(
        run_id=result.run_id,
        agent_name=result.agent_name,
        provider=str(getattr(agent.model, "provider", "")),
        model_id=str(getattr(agent.model, "model_id", "")),
        status=status,
        session_id=result.session.id,
        parent_run_id=parent_run_id,
        idempotency_key=idempotency_key,
        started_at_ms=started_at_ms,
        updated_at_ms=finished_at_ms,
        finished_at_ms=finished_at_ms,
        current_step=len(result.steps),
        steps=steps,
        child_runs=_child_runs_from_tool_results(result.tool_results, result.run_id),
        tool_results=list(result.tool_results),
        usage=result.usage,
        output_text=result.text,
        finish_reason=result.finish_reason,
        error=error,
        metadata={"orchestration_path": list(result.orchestration_path)},
    )
    return state


def _segment_text(result: GenerateTextOutput) -> str:
    if result.steps and result.steps[-1].response.text:
        return result.steps[-1].response.text or ""
    return result.text


def _segment_finish_reason(result: GenerateTextOutput) -> FinishReason | None:
    if result.steps:
        return result.steps[-1].response.finish_reason
    return result.finish_reason


def _segment_provider_finish_reason(result: GenerateTextOutput) -> str | None:
    if result.steps:
        return result.steps[-1].response.provider_finish_reason
    return result.provider_finish_reason


def _merge_usage(usages: list[TokenUsage | None]) -> TokenUsage | None:
    present = [usage for usage in usages if usage is not None]
    if not present:
        return None
    input_tokens = (
        sum(usage.input_tokens for usage in present if usage.input_tokens is not None)
        if all(usage.input_tokens is not None for usage in present)
        else None
    )
    output_tokens = (
        sum(usage.output_tokens for usage in present if usage.output_tokens is not None)
        if all(usage.output_tokens is not None for usage in present)
        else None
    )
    total_tokens = (
        sum(usage.total_tokens for usage in present if usage.total_tokens is not None)
        if all(usage.total_tokens is not None for usage in present)
        else None
    )
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)


def _response_messages(step: GenerateTextStep) -> list[ModelMessage]:
    if step.response.messages:
        return list(step.response.messages)
    if step.response.message is not None:
        return [step.response.message]
    if step.response.text:
        return [create_text_message("assistant", step.response.text)]
    return []


async def _save_checkpoints(
    *,
    checkpoint_store: AgentCheckpointStore | None,
    result: GenerateTextOutput,
    run_id: str,
    session_id: str,
    agent_name: str,
    emit: Callable[[AgentEvent], Awaitable[None]] | None = None,
    trace: AgentTrace | None = None,
) -> None:
    if checkpoint_store is None:
        return
    for index, step in enumerate(result.steps, start=1):
        checkpoint = AgentCheckpoint(
            run_id=run_id,
            session_id=session_id,
            agent_name=agent_name,
            step_index=index,
            request=step.request,
            response=step.response,
            saved_at_ms=_now_ms(),
            is_final=index == len(result.steps),
        )
        await checkpoint_store.save(checkpoint)
        if trace is not None:
            trace.checkpoint_count += 1
        if emit is not None:
            await emit(AgentCheckpointEvent(checkpoint=checkpoint))


class AgentRegistry:
    def __init__(self, agents: dict[str, Agent] | None = None) -> None:
        self._agents: dict[str, Agent] = dict(agents or {})

    def register(self, agent: Agent) -> Agent:
        self._agents[agent.name] = agent
        return agent

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)


@dataclass
class _Broadcast:
    history: list[AgentEvent]
    done: bool = False
    subscribers: list[asyncio.Queue[AgentEvent | None]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.subscribers = []

    async def publish(self, event: AgentEvent) -> None:
        self.history.append(event)
        for queue in list(self.subscribers):
            await queue.put(event)

    async def close(self) -> None:
        self.done = True
        for queue in list(self.subscribers):
            await queue.put(None)

    def stream(self) -> AsyncIterable[AgentEvent]:
        async def generator() -> AsyncIterable[AgentEvent]:
            queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
            cursor = 0
            self.subscribers.append(queue)
            try:
                while True:
                    while cursor < len(self.history):
                        event = self.history[cursor]
                        cursor += 1
                        yield event
                    if self.done:
                        return
                    item = await queue.get()
                    if item is None:
                        return
            finally:
                if queue in self.subscribers:
                    self.subscribers.remove(queue)

        return generator()


class AgentObserver(Protocol):
    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Any: ...


class AgentRuntime:
    def __init__(
        self,
        *,
        registry: AgentRegistry | None = None,
        observer: AgentObserver | None = None,
    ) -> None:
        self._registry = registry or AgentRegistry()
        self._observer = observer

    async def run(
        self,
        *,
        agent: Agent,
        session: AgentSession | None = None,
        prompt: str | None = None,
        messages: list[ModelMessage] | None = None,
        tools: ToolSet | ToolRegistry | None = None,
        skills: SkillSet | SkillRegistry | None = None,
        tool_choice: str | ToolChoiceName | None = None,
        tool_execution: ToolExecutionOptions | None = None,
        max_steps: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: ReasoningConfig | None = None,
        provider_options: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
        max_retries: int | None = None,
        retry_backoff_ms: int | None = None,
        stop_on_handoff: bool = False,
        emit: Callable[[AgentEvent], Awaitable[None]] | None = None,
        resumed_from_checkpoint: AgentCheckpoint | None = None,
        live_stream: bool = False,
        parent_run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> AgentRunResult:
        resolved_session = session or create_agent_session()
        if agent.memory is not None and not resolved_session.messages and resolved_session.summary is None:
            state = await agent.memory.load(resolved_session.id)
            resolved_session.messages = list(state.messages)
            resolved_session.summary = state.summary
            resolved_session.metadata = {**state.metadata, **resolved_session.metadata}

        run_id = _new_id("run")
        started_at_ms = _now_ms()
        if idempotency_key and agent.run_store is not None:
            existing_state = await agent.run_store.find_by_idempotency_key(idempotency_key)
            if existing_state is not None:
                return AgentRunResult(
                    run_id=existing_state.run_id,
                    agent_name=existing_state.agent_name,
                    session=resolved_session,
                    text=existing_state.output_text,
                    finish_reason=existing_state.finish_reason,
                    usage=existing_state.usage,
                    orchestration_path=[existing_state.agent_name],
                    state=existing_state,
                )
        trace = AgentTrace(
            run_id=run_id,
            session_id=resolved_session.id,
            agent_name=agent.name,
            started_at_ms=started_at_ms,
            orchestration_path=[agent.name],
        )

        async def publish(event: AgentEvent) -> None:
            trace.events.append(event)
            if emit is not None:
                await emit(event)

        await publish(AgentRunStartEvent(run_id=run_id, session_id=resolved_session.id, agent_name=agent.name))
        if agent.run_store is not None:
            await agent.run_store.save(
                AgentRunState(
                    run_id=run_id,
                    agent_name=agent.name,
                    provider=str(getattr(agent.model, "provider", "")),
                    model_id=str(getattr(agent.model, "model_id", "")),
                    session_id=resolved_session.id,
                    parent_run_id=parent_run_id,
                    idempotency_key=idempotency_key,
                    started_at_ms=started_at_ms,
                    updated_at_ms=started_at_ms,
                    metadata={"orchestration_path": [agent.name]},
                )
            )

        current_agent = agent
        current_prompt = prompt
        current_messages = messages
        handoff_depth = 0
        try:
            while True:
                trace.segments.append(AgentTraceSegment(agent_name=current_agent.name, started_at_ms=_now_ms()))
                await publish(AgentDelegationStartEvent(agent_name=current_agent.name, handoff_depth=handoff_depth))
                segment_result = await self._run_single(
                    agent=current_agent,
                    session=resolved_session,
                    run_id=run_id,
                    trace=trace,
                    prompt=current_prompt,
                    messages=current_messages,
                    tools=tools,
                    skills=skills,
                    tool_choice=tool_choice,
                    tool_execution=tool_execution,
                    max_steps=max_steps,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning=reasoning,
                    provider_options=provider_options,
                    timeout_ms=timeout_ms,
                    max_retries=max_retries,
                    retry_backoff_ms=retry_backoff_ms,
                    emit=publish,
                    live_stream=live_stream,
                )
                trace.segments[-1].finished_at_ms = _now_ms()
                await publish(
                    AgentDelegationFinishEvent(
                        agent_name=current_agent.name,
                        handoff_depth=handoff_depth,
                        finish_reason=segment_result.finish_reason,
                    )
                )
                if segment_result.handoff is None or stop_on_handoff:
                    final_handoff = segment_result.handoff if stop_on_handoff else None
                    await publish(
                        AgentFinishEvent(
                            run_id=run_id,
                            session_id=resolved_session.id,
                            text=segment_result.text,
                            finish_reason=segment_result.finish_reason,
                        )
                    )
                    trace.finished_at_ms = _now_ms()
                    output = AgentRunResult(
                        run_id=run_id,
                        agent_name=segment_result.agent_name,
                        session=resolved_session,
                        text=segment_result.text,
                        finish_reason=segment_result.finish_reason,
                        provider_finish_reason=segment_result.provider_finish_reason,
                        usage=segment_result.usage,
                        steps=segment_result.steps,
                        messages=segment_result.messages,
                        tool_results=segment_result.tool_results,
                        artifacts=segment_result.artifacts,
                        trace=trace,
                        handoff=final_handoff,
                        orchestration_path=list(trace.orchestration_path),
                        resumed_from_checkpoint=resumed_from_checkpoint,
                    )
                    run_state = _agent_run_state_from_result(
                        result=output,
                        agent=agent,
                        parent_run_id=parent_run_id,
                        idempotency_key=idempotency_key,
                        started_at_ms=started_at_ms,
                        finished_at_ms=trace.finished_at_ms,
                    )
                    output.state = run_state
                    if agent.run_store is not None:
                        await agent.run_store.save(run_state)
                    return output

                handoff = segment_result.handoff
                await publish(AgentHandoffRequestedEvent(handoff=handoff))
                trace.handoff_count += 1
                if current_agent.run_limits.max_handoffs is not None and trace.handoff_count > current_agent.run_limits.max_handoffs:
                    raise RuntimeError(f'Agent exceeded max handoffs ({current_agent.run_limits.max_handoffs}).')
                next_agent = current_agent.subagents.get(handoff.target_agent) or self._registry.get(handoff.target_agent)
                if next_agent is None:
                    await publish(
                        AgentHandoffFailedEvent(
                            source_agent=current_agent.name,
                            target_agent=handoff.target_agent,
                            reason="Unknown handoff target.",
                        )
                    )
                    raise RuntimeError(f'Unknown handoff target "{handoff.target_agent}".')
                await publish(
                    AgentHandoffResolvedEvent(
                        source_agent=current_agent.name,
                        target_agent=next_agent.name,
                    )
                )
                await publish(AgentHandoffEvent(handoff=handoff))
                current_agent = next_agent
                trace.orchestration_path.append(next_agent.name)
                current_prompt = handoff.input or f"Continue the delegated task from {trace.orchestration_path[-2]}."
                current_messages = None
                handoff_depth += 1
        except Exception as error:
            await publish(AgentErrorEvent(error=error))
            trace.finished_at_ms = _now_ms()
            if agent.run_store is not None:
                await agent.run_store.save(
                    AgentRunState(
                        run_id=run_id,
                        agent_name=agent.name,
                        provider=str(getattr(agent.model, "provider", "")),
                        model_id=str(getattr(agent.model, "model_id", "")),
                        status="failed",
                        session_id=resolved_session.id,
                        parent_run_id=parent_run_id,
                        idempotency_key=idempotency_key,
                        started_at_ms=started_at_ms,
                        updated_at_ms=trace.finished_at_ms,
                        finished_at_ms=trace.finished_at_ms,
                        error=str(error),
                        metadata={"orchestration_path": list(trace.orchestration_path)},
                    )
                )
            raise

    async def _run_input_guardrails(
        self,
        *,
        agent: Agent,
        run_id: str,
        session_id: str,
        prompt: str | None,
        messages: list[ModelMessage],
        context: AgentContext,
        trace: AgentTrace,
        emit: Callable[[AgentEvent], Awaitable[None]],
    ) -> None:
        request = InputGuardrailRequest(
            run_id=run_id,
            session_id=session_id,
            agent_name=agent.name,
            prompt=prompt,
            messages=list(messages),
            context=context,
        )
        for guardrail in agent.input_guardrails:
            name = _guardrail_name(guardrail)
            span = self._start_span(
                "zhivex.agent.guardrail",
                {
                    "guardrail.name": name,
                    "guardrail.stage": "input",
                    "agent.name": agent.name,
                    "run.id": run_id,
                },
            )
            try:
                outcome = _normalize_guardrail_result(await _maybe_await(guardrail(request)))
            except Exception as error:
                self._finish_span(span, error=error)
                raise
            self._finish_span(
                span,
                attributes={
                    "guardrail.triggered": outcome.tripwire_triggered,
                },
            )
            await emit(
                AgentGuardrailEvent(
                    stage="input",
                    guardrail_name=name,
                    triggered=outcome.tripwire_triggered,
                    reason=outcome.reason,
                    metadata=dict(outcome.metadata),
                )
            )
            if outcome.tripwire_triggered:
                trace.guardrail_trigger_count += 1
                raise GuardrailTripwireTriggered(
                    stage="input",
                    guardrail_name=name,
                    reason=outcome.reason,
                    metadata=outcome.metadata,
                )

    async def _run_output_guardrails(
        self,
        *,
        agent: Agent,
        run_id: str,
        session_id: str,
        result: GenerateTextOutput | None,
        text: str,
        messages: list[ModelMessage],
        context: AgentContext,
        trace: AgentTrace,
        emit: Callable[[AgentEvent], Awaitable[None]],
    ) -> None:
        request = OutputGuardrailRequest(
            run_id=run_id,
            session_id=session_id,
            agent_name=agent.name,
            text=text,
            messages=list(messages),
            result=result,
            context=context,
        )
        for guardrail in agent.output_guardrails:
            name = _guardrail_name(guardrail)
            span = self._start_span(
                "zhivex.agent.guardrail",
                {
                    "guardrail.name": name,
                    "guardrail.stage": "output",
                    "agent.name": agent.name,
                    "run.id": run_id,
                },
            )
            try:
                outcome = _normalize_guardrail_result(await _maybe_await(guardrail(request)))
            except Exception as error:
                self._finish_span(span, error=error)
                raise
            self._finish_span(
                span,
                attributes={
                    "guardrail.triggered": outcome.tripwire_triggered,
                },
            )
            await emit(
                AgentGuardrailEvent(
                    stage="output",
                    guardrail_name=name,
                    triggered=outcome.tripwire_triggered,
                    reason=outcome.reason,
                    metadata=dict(outcome.metadata),
                )
            )
            if outcome.tripwire_triggered:
                trace.guardrail_trigger_count += 1
                raise GuardrailTripwireTriggered(
                    stage="output",
                    guardrail_name=name,
                    reason=outcome.reason,
                    metadata=outcome.metadata,
                )

    async def _run_single(
        self,
        *,
        agent: Agent,
        session: AgentSession,
        run_id: str,
        trace: AgentTrace,
        prompt: str | None,
        messages: list[ModelMessage] | None,
        tools: ToolSet | ToolRegistry | None,
        skills: SkillSet | SkillRegistry | None,
        tool_choice: str | ToolChoiceName | None,
        tool_execution: ToolExecutionOptions | None,
        max_steps: int | None,
        temperature: float | None,
        max_tokens: int | None,
        reasoning: ReasoningConfig | None,
        provider_options: dict[str, Any] | None,
        timeout_ms: int | None,
        max_retries: int | None,
        retry_backoff_ms: int | None,
        emit: Callable[[AgentEvent], Awaitable[None]],
        live_stream: bool,
    ) -> AgentRunResult:
        active_skill_activations, skipped_skills, skill_tools = await _select_active_skills(
            _resolve_skill_registry(agent, skills),
            agent=agent,
            session=session,
            prompt=prompt,
            messages=messages,
        )
        await _emit_skill_events(active_skills=active_skill_activations, skipped_skills=skipped_skills, emit=emit)
        built_messages = _build_run_messages(
            agent=agent,
            session=session,
            prompt=prompt,
            messages=messages,
            active_skills=[item.skill for item in active_skill_activations],
        )
        _persist_active_skills(session, active_skill_activations)
        context = AgentContext(
            run_id=run_id,
            session_id=session.id,
            agent_name=agent.name,
            memory_summary=session.summary,
            metadata={
                **dict(agent.metadata),
                "skills": [item.skill.name for item in active_skill_activations],
            },
            handoff_path=list(trace.orchestration_path),
        )
        await self._run_input_guardrails(
            agent=agent,
            run_id=run_id,
            session_id=session.id,
            prompt=prompt,
            messages=built_messages,
            context=context,
            trace=trace,
            emit=emit,
        )
        registry = _resolve_tool_registry(agent, tools)
        if agent.subagents:
            registry = registry.merge(
                {
                    name: create_subagent_tool(name=name, agent=subagent, parent_run_id=run_id)
                    for name, subagent in agent.subagents.items()
                }
            )
        if skill_tools:
            registry = registry.merge(skill_tools)
        merged_tools = self._wrap_agent_tools(
            agent=agent,
            registry=registry,
            run_id=run_id,
            session_id=session.id,
            trace=trace,
            started_at_ms=trace.started_at_ms,
            context=context,
            emit=emit,
        )
        span = self._start_span(
            "zhivex.agent.model",
            {
                "agent.name": agent.name,
                "run.id": run_id,
                "session.id": session.id,
                "orchestration.depth": len(trace.orchestration_path) - 1,
            },
        )
        buffer_live_text = live_stream and agent.model.capabilities.streaming and bool(agent.output_guardrails)
        buffered_text_deltas: list[str] = []
        accumulated_steps: list[GenerateTextStep] = []
        accumulated_tool_results: list[ToolExecutionResult] = []
        conversation_messages = list(built_messages)
        persisted_run_messages: list[ModelMessage] = []
        remaining_steps = _effective_max_steps(agent.run_limits, max_steps)

        async def resolve_provider_managed_approval(
            approval: _ProviderManagedApproval,
        ) -> ModelMessage:
            if agent.approval_policy is None:
                raise RuntimeError(
                    "Provider-managed approvals require an approval_policy on the agent."
                )
            trace.approval_count += 1
            request = ToolApprovalRequest(
                run_id=run_id,
                session_id=session.id,
                agent_name=agent.name,
                tool_name=approval.tool_name,
                tool_input=approval.tool_input,
                tool_permissions=[],
                tool_source="hosted",
                tool_metadata={
                    "provider": approval.provider,
                    "server_label": approval.server_label,
                    "hosted_tool_class": "remote-mcp",
                    "provider_event_type": "mcp_approval_request",
                    "raw_provider_payload": approval.raw_payload,
                },
                context=context,
                handoff_path=list(trace.orchestration_path),
            )
            decision = _normalize_approval_decision(await _maybe_await(agent.approval_policy(request)))
            await emit(
                AgentToolApprovalEvent(
                    tool_name=approval.tool_name,
                    tool_input=approval.tool_input,
                    approved=decision.approved,
                    reason=decision.reason,
                    provider=approval.provider,
                    provider_managed=True,
                    approval_request_id=approval.approval_request_id,
                    tool_source="hosted",
                    metadata={
                        "provider": approval.provider,
                        "server_label": approval.server_label,
                        "hosted_tool_class": "remote-mcp",
                        "provider_event_type": "mcp_approval_request",
                        "raw_provider_payload": approval.raw_payload,
                    },
                )
            )
            return _provider_managed_approval_response_message(
                approval,
                approved=decision.approved,
                reason=decision.reason,
            )

        try:
            emitted_live_text = False
            emitted_live_tool_events = False
            while True:
                if remaining_steps is not None and remaining_steps <= 0:
                    raise RuntimeError("Agent run exceeded max steps while handling provider-managed approvals.")
                pending_provider_responses: list[ModelMessage] = []
                handled_provider_approvals: set[str] = set()
                handled_provider_tool_events: set[str] = set()
                if live_stream and agent.model.capabilities.streaming:

                    async def handle_stream_event(event: Any) -> None:
                        nonlocal emitted_live_text, emitted_live_tool_events
                        if isinstance(event, StreamTextDeltaEvent):
                            if buffer_live_text:
                                buffered_text_deltas.append(event.text_delta)
                            else:
                                emitted_live_text = True
                                await emit(AgentTextDeltaEvent(text_delta=event.text_delta))
                        elif isinstance(event, StreamToolCallEvent):
                            emitted_live_tool_events = True
                            await emit(AgentToolCallEvent(tool_call=event.tool_call))
                        elif isinstance(event, StreamToolResultEvent):
                            emitted_live_tool_events = True
                            await emit(AgentToolResultEvent(tool_result=event.tool_result))
                        elif isinstance(event, StreamProviderDataEvent):
                            provider_part = provider_data_part(event.provider, event.data)
                            provider_tool_event = _parse_provider_managed_tool_trace_part(provider_part)
                            if (
                                provider_tool_event is not None
                                and provider_tool_event.event_key not in handled_provider_tool_events
                            ):
                                handled_provider_tool_events.add(provider_tool_event.event_key)
                                emitted_live_tool_events = True
                                await emit(AgentToolCallEvent(tool_call=provider_tool_event.tool_call))
                            approval = _parse_provider_managed_approval_part(provider_part)
                            if approval is None or approval.approval_request_id in handled_provider_approvals:
                                return
                            handled_provider_approvals.add(approval.approval_request_id)
                            pending_provider_responses.append(await resolve_provider_managed_approval(approval))

                    streamed = stream_text(
                        model=cast(LanguageModel, agent.model),
                        messages=conversation_messages,
                        tools=merged_tools or None,
                        tool_choice=cast(Any, tool_choice),
                        tool_execution=tool_execution,
                        max_steps=remaining_steps,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        reasoning=reasoning,
                        provider_options=provider_options,
                        timeout_ms=_effective_timeout_ms(agent.run_limits, timeout_ms),
                        max_retries=max_retries,
                        retry_backoff_ms=retry_backoff_ms,
                        on_event=handle_stream_event,
                    )
                    result = await streamed.collect()
                else:
                    result = await generate_text(
                        model=cast(LanguageModel, agent.model),
                        messages=conversation_messages,
                        tools=merged_tools or None,
                        tool_choice=cast(Any, tool_choice),
                        tool_execution=tool_execution,
                        max_steps=remaining_steps,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        reasoning=reasoning,
                        provider_options=provider_options,
                        timeout_ms=_effective_timeout_ms(agent.run_limits, timeout_ms),
                        max_retries=max_retries,
                        retry_backoff_ms=retry_backoff_ms,
                    )

                accumulated_steps.extend(result.steps)
                accumulated_tool_results.extend(result.tool_results)
                if remaining_steps is not None:
                    remaining_steps -= max(1, len(result.steps))

                new_response_messages: list[ModelMessage] = []
                for step in result.steps:
                    response_messages = _response_messages(step)
                    if response_messages:
                        new_response_messages.extend(response_messages)
                        conversation_messages.extend(response_messages)
                        persisted_run_messages.extend(response_messages)
                for tool_result in result.tool_results:
                    tool_message = ModelMessage(role="tool", parts=[tool_result_part(tool_result)])
                    conversation_messages.append(tool_message)
                    persisted_run_messages.append(tool_message)

                for message in new_response_messages:
                    for part in message.parts:
                        provider_tool_event = _parse_provider_managed_tool_trace_part(part)
                        if provider_tool_event is not None and provider_tool_event.event_key not in handled_provider_tool_events:
                            handled_provider_tool_events.add(provider_tool_event.event_key)
                            await emit(AgentToolCallEvent(tool_call=provider_tool_event.tool_call))
                        approval = _parse_provider_managed_approval_part(part)
                        if approval is None or approval.approval_request_id in handled_provider_approvals:
                            continue
                        handled_provider_approvals.add(approval.approval_request_id)
                        pending_provider_responses.append(await resolve_provider_managed_approval(approval))

                if pending_provider_responses:
                    conversation_messages.extend(pending_provider_responses)
                    persisted_run_messages.extend(pending_provider_responses)
                    continue
                break
        except Exception as error:
            self._finish_span(span, error=error)
            raise
        self._finish_span(span, attributes={"finish.reason": result.finish_reason})

        if not emitted_live_tool_events:
            for tool_call in _extract_tool_calls_from_steps(accumulated_steps):
                await emit(AgentToolCallEvent(tool_call=tool_call))
            for tool_result in _extract_provider_tool_results_from_steps(accumulated_steps):
                await emit(AgentToolResultEvent(tool_result=tool_result))
        segment_text = _segment_text(result)
        segment_finish_reason = _segment_finish_reason(result)
        segment_provider_finish_reason = _segment_provider_finish_reason(result)
        if segment_text and not emitted_live_text and not buffer_live_text:
            await emit(AgentTextDeltaEvent(text_delta=segment_text))
        if not emitted_live_tool_events:
            for tool_result in accumulated_tool_results:
                await emit(AgentToolResultEvent(tool_result=tool_result))
        await self._run_output_guardrails(
            agent=agent,
            run_id=run_id,
            session_id=session.id,
            result=result,
            text=segment_text,
            messages=_assistant_messages_from_result(result),
            context=context,
            trace=trace,
            emit=emit,
        )
        if buffer_live_text:
            if buffered_text_deltas:
                for text_delta in buffered_text_deltas:
                    await emit(AgentTextDeltaEvent(text_delta=text_delta))
            elif segment_text:
                await emit(AgentTextDeltaEvent(text_delta=segment_text))
        transcript = list(session.messages)
        if messages is not None:
            transcript.extend(messages)
        elif prompt is not None:
            transcript.append(create_text_message("user", prompt))
        transcript.extend(persisted_run_messages)
        session.messages = _strip_runtime_system_messages(transcript, agent.instructions)
        if agent.memory is not None and _should_refresh_summary(agent.memory, session):
            summary_span = self._start_span(
                "zhivex.agent.summary",
                {"agent.name": agent.name, "run.id": run_id, "session.id": session.id},
            )
            session.summary = await agent.memory.summarize(
                session_id=session.id,
                state=AgentMemoryState(
                    messages=list(session.messages),
                    summary=session.summary,
                    metadata={**dict(session.metadata), "state": dict(session.state)},
                ),
                agent=agent,
            )
            self._finish_span(summary_span, attributes={"summary.updated": bool(session.summary)})
            await emit(AgentSummaryUpdateEvent(summary=session.summary))
        if agent.memory is not None:
            await agent.memory.save(
                session.id,
                AgentMemoryState(
                    messages=list(session.messages),
                    summary=session.summary,
                    metadata={**dict(session.metadata), "state": dict(session.state)},
                ),
            )

        await _save_checkpoints(
            checkpoint_store=agent.checkpoint_store,
            result=GenerateTextOutput(
                text=segment_text,
                finish_reason=segment_finish_reason,
                provider_finish_reason=segment_provider_finish_reason,
                usage=_merge_usage([step.response.usage for step in accumulated_steps]),
                steps=accumulated_steps,
                messages=conversation_messages,
                tool_results=accumulated_tool_results,
            ),
            run_id=run_id,
            session_id=session.id,
            agent_name=agent.name,
            emit=emit,
            trace=trace,
        )
        artifacts = _extract_skill_artifacts(accumulated_tool_results)
        if artifacts:
            session.metadata = {
                **session.metadata,
                "skill_artifacts": [
                    {
                        "name": artifact.name,
                        "path": artifact.path,
                        "media_type": artifact.media_type,
                        "role": artifact.role,
                        "description": artifact.description,
                        "metadata": dict(artifact.metadata),
                    }
                    for artifact in artifacts
                ],
            }
        handoff = _detect_handoff(accumulated_tool_results)
        return AgentRunResult(
            run_id=run_id,
            agent_name=agent.name,
            session=session,
            text=segment_text,
            finish_reason=segment_finish_reason,
            provider_finish_reason=segment_provider_finish_reason,
            usage=_merge_usage([step.response.usage for step in accumulated_steps]),
            steps=accumulated_steps,
            messages=conversation_messages,
            tool_results=accumulated_tool_results,
            artifacts=artifacts,
            trace=trace,
            handoff=handoff,
            orchestration_path=list(trace.orchestration_path),
        )

    def _wrap_agent_tools(
        self,
        *,
        agent: Agent,
        registry: ToolRegistry,
        run_id: str,
        session_id: str,
        trace: AgentTrace,
        started_at_ms: int,
        context: AgentContext,
        emit: Callable[[AgentEvent], Awaitable[None]],
    ) -> ToolSet:
        wrapped: ToolSet = {}
        tool_limit = agent.run_limits.max_tool_calls

        for tool_name, definition in registry.items():
            if not is_callable_tool_definition(definition):
                continue
            callable_definition = definition

            async def execute(
                input: Any,
                call_context: ToolExecutionContext | None = None,
                *,
                _tool_name: str = tool_name,
                _definition: ToolDefinition = callable_definition,
            ) -> Any:
                if agent.run_limits.max_wall_time_ms is not None and _now_ms() - started_at_ms > agent.run_limits.max_wall_time_ms:
                    raise RuntimeError("Agent run exceeded max wall time.")

                trace.tool_call_count += 1
                if tool_limit is not None and trace.tool_call_count > tool_limit:
                    raise RuntimeError(f'Agent exceeded max tool calls ({tool_limit}).')

                request = ToolApprovalRequest(
                    run_id=run_id,
                    session_id=session_id,
                    agent_name=agent.name,
                    tool_name=_tool_name,
                    tool_input=input,
                    tool_permissions=list(_definition.permissions),
                    tool_source=_definition.source,
                    tool_metadata=dict(_definition.metadata),
                    context=context,
                    handoff_path=list(trace.orchestration_path),
                )
                decision = ApprovalDecision(approved=True)
                if _definition.requires_approval or (
                    _definition.requires_approval is None and agent.approval_policy is not None
                ):
                    trace.approval_count += 1
                    if agent.approval_policy is not None:
                        decision = _normalize_approval_decision(await _maybe_await(agent.approval_policy(request)))
                    await emit(
                        AgentToolApprovalEvent(
                            tool_name=_tool_name,
                            tool_input=input,
                            approved=decision.approved,
                            reason=decision.reason,
                        )
                    )
                if not decision.approved:
                    raise RuntimeError(decision.reason or f'Tool "{_tool_name}" denied by approval policy.')

                tool_context = ToolExecutionContext(
                    tool_name=_tool_name,
                    tool_call_id=call_context.tool_call_id if call_context is not None else "",
                    run_id=run_id,
                    session_id=session_id,
                    agent_name=agent.name,
                    memory_summary=context.memory_summary,
                    permissions=list(_definition.permissions),
                    source=_definition.source,
                    metadata={**context.metadata, **_definition.metadata},
                    handoff_path=list(trace.orchestration_path),
                )
                span = self._start_span(
                    "zhivex.agent.tool",
                    {
                        "tool.name": _tool_name,
                        "tool.source": _definition.source,
                        "agent.name": agent.name,
                        "run.id": run_id,
                    },
                )
                skill_name = str(_definition.metadata.get("skill_name") or "")
                skill_entrypoint = str(_definition.metadata.get("skill_entrypoint") or "")
                if skill_name and skill_entrypoint:
                    await emit(AgentSkillExecutionStartEvent(skill_name=skill_name, entrypoint=skill_entrypoint))
                try:
                    result = await registry.execute(_definition, input, tool_context)
                except Exception as error:
                    if skill_name and skill_entrypoint:
                        await emit(AgentSkillExecutionFinishEvent(skill_name=skill_name, entrypoint=skill_entrypoint, ok=False))
                    self._finish_span(span, error=error)
                    raise
                if skill_name and skill_entrypoint:
                    await emit(AgentSkillExecutionFinishEvent(skill_name=skill_name, entrypoint=skill_entrypoint, ok=True))
                    if isinstance(result, dict):
                        for item in list(result.get("artifacts") or []):
                            if isinstance(item, dict):
                                artifact_path = str(item.get("path") or "").strip()
                                if artifact_path:
                                    artifact_metadata = item.get("metadata")
                                    await emit(
                                        AgentSkillArtifactCreatedEvent(
                                            skill_name=skill_name,
                                            entrypoint=skill_entrypoint,
                                            artifact=SkillArtifact(
                                                name=str(item.get("name") or Path(artifact_path).name),
                                                path=artifact_path,
                                                media_type=str(item.get("media_type") or "") or None,
                                                role=str(item.get("role") or "primary"),  # type: ignore[arg-type]
                                                description=str(item.get("description") or "") or None,
                                                metadata=cast(dict[str, Any], artifact_metadata)
                                                if isinstance(artifact_metadata, dict)
                                                else {},
                                            ),
                                        )
                                    )
                self._finish_span(span)
                return result

            wrapped[tool_name] = ToolDefinition(
                name=definition.name,
                description=definition.description,
                schema=definition.schema,
                execute=execute,
                tags=list(definition.tags),
                requires_approval=definition.requires_approval,
                permissions=list(definition.permissions),
                source=definition.source,
                metadata=dict(definition.metadata),
                supports_streaming=definition.supports_streaming,
                remote_config=definition.remote_config,
                mcp_config=definition.mcp_config,
            )

        return wrapped

    def _start_span(self, name: str, attributes: dict[str, Any]) -> Any:
        if self._observer is None:
            return None
        return self._observer.start_span(name, attributes)

    def _finish_span(self, span: Any, *, attributes: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        if span is None:
            return
        span.end(attributes=attributes, error=error)


class AgentStreamResult:
    def __init__(self, runner: asyncio.Task[AgentRunResult], broadcast: _Broadcast) -> None:
        self._runner = runner
        self._broadcast = broadcast

    def event_stream(self) -> AsyncIterable[AgentEvent]:
        return self._broadcast.stream()

    def text_stream(self) -> AsyncIterable[str]:
        async def generator() -> AsyncIterable[str]:
            async for event in self._broadcast.stream():
                if isinstance(event, AgentTextDeltaEvent):
                    yield event.text_delta

        return generator()

    async def collect(self) -> AgentRunResult:
        return await self._runner


AgentLiveEvent: TypeAlias = AgentEvent | RealtimeEvent


@dataclass
class _LiveBroadcast:
    history: list[AgentLiveEvent]
    done: bool = False
    subscribers: list[asyncio.Queue[AgentLiveEvent | None]] | None = None

    def __post_init__(self) -> None:
        self.subscribers = []

    async def publish(self, event: AgentLiveEvent) -> None:
        self.history.append(event)
        for queue in list(self.subscribers or []):
            await queue.put(event)

    async def close(self) -> None:
        self.done = True
        for queue in list(self.subscribers or []):
            await queue.put(None)

    def stream(self) -> AsyncIterable[AgentLiveEvent]:
        async def generator() -> AsyncIterable[AgentLiveEvent]:
            queue: asyncio.Queue[AgentLiveEvent | None] = asyncio.Queue()
            cursor = 0
            self.subscribers = self.subscribers or []
            self.subscribers.append(queue)
            try:
                while True:
                    while cursor < len(self.history):
                        event = self.history[cursor]
                        cursor += 1
                        yield event
                    if self.done:
                        return
                    item = await queue.get()
                    if item is None:
                        return
            finally:
                if self.subscribers and queue in self.subscribers:
                    self.subscribers.remove(queue)

        return generator()


class LiveAgentStreamResult:
    def __init__(self, runner: asyncio.Task[AgentRunResult], broadcast: _LiveBroadcast, live_session: asyncio.Future[Any]) -> None:
        self._runner = runner
        self._broadcast = broadcast
        self._live_session = live_session

    def event_stream(self) -> AsyncIterable[AgentLiveEvent]:
        return self._broadcast.stream()

    async def send_audio(self, frame: AudioFrame) -> None:
        await (await self._live_session).send_audio(frame)

    async def send_text(self, text: str) -> None:
        await (await self._live_session).send_text(text)

    async def update(
        self,
        *,
        instructions: str | None = None,
        voice: str | None = None,
        tools: ToolSet | None = None,
        tool_choice: str | ToolChoiceName | None = None,
        turn_detection: dict[str, Any] | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> None:
        await (await self._live_session).update(
            instructions=instructions,
            voice=voice,
            tools=tools,
            tool_choice=tool_choice,
            turn_detection=turn_detection,
            provider_options=provider_options,
        )

    async def aclose(self) -> None:
        await (await self._live_session).aclose()

    async def collect(self) -> AgentRunResult:
        return await self._runner


def create_subagent_tool(
    *,
    name: str,
    agent: Agent,
    parent_run_id: str | None = None,
    description: str | None = None,
) -> ToolDefinition:
    async def execute(input: Any) -> JsonValue:
        prompt = input.get("prompt") if isinstance(input, dict) else str(input)
        result = await run_agent(agent=agent, prompt=str(prompt or ""), parent_run_id=parent_run_id)
        child_state = result.state
        if child_state is None:
            child_state = _agent_run_state_from_result(
                result=result,
                agent=agent,
                parent_run_id=parent_run_id,
                idempotency_key=None,
                started_at_ms=result.trace.started_at_ms if result.trace else _now_ms(),
                finished_at_ms=result.trace.finished_at_ms if result.trace and result.trace.finished_at_ms else _now_ms(),
            )
        return {
            "text": result.text,
            "child_run": asdict(agent_child_run_from_state(child_state, tool_name=name)),
        }

    return ToolDefinition(
        name=name,
        description=description or f"Run the {agent.name} subagent.",
        schema={
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
        execute=execute,
        metadata={"type": "subagent", "agent_name": agent.name, "parent_run_id": parent_run_id},
    )


def prepare_subagents_for_agent(agent: Agent) -> Agent:
    if not agent.subagents:
        return agent
    registry = _resolve_tool_registry(agent, None)
    registry = registry.merge(
        {name: create_subagent_tool(name=name, agent=subagent) for name, subagent in agent.subagents.items()}
    )
    return replace(agent, tools=registry)


@dataclass(slots=True)
class AgentGroupMember:
    name: str
    agent: Agent
    prompt: str | None = None


@dataclass(slots=True)
class AgentGroupMemberResult:
    name: str
    output: AgentRunResult | None = None
    error: Exception | None = None


@dataclass(slots=True)
class AgentGroupRunResult:
    parent_run_id: str | None
    outputs: list[AgentGroupMemberResult]


async def run_agent_group(
    members: list[AgentGroupMember],
    *,
    prompt: str | None = None,
    parent_run_id: str | None = None,
) -> AgentGroupRunResult:
    async def run_member(member: AgentGroupMember) -> AgentGroupMemberResult:
        try:
            output = await run_agent(
                agent=member.agent,
                prompt=member.prompt if member.prompt is not None else prompt,
                parent_run_id=parent_run_id,
            )
            return AgentGroupMemberResult(name=member.name, output=output)
        except Exception as error:
            return AgentGroupMemberResult(name=member.name, error=error)

    outputs = await asyncio.gather(*(run_member(member) for member in members))
    return AgentGroupRunResult(parent_run_id=parent_run_id, outputs=list(outputs))


def run_agent(
    *,
    agent: Agent,
    session: AgentSession | None = None,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    tools: ToolSet | ToolRegistry | None = None,
    skills: SkillSet | SkillRegistry | None = None,
    tool_choice: str | ToolChoiceName | None = None,
    tool_execution: ToolExecutionOptions | None = None,
    max_steps: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning: ReasoningConfig | None = None,
    provider_options: dict[str, Any] | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
    stop_on_handoff: bool = False,
    runtime: AgentRuntime | None = None,
    registry: AgentRegistry | None = None,
    observer: AgentObserver | None = None,
    parent_run_id: str | None = None,
    idempotency_key: str | None = None,
) -> Awaitable[AgentRunResult]:
    resolved_runtime = runtime or AgentRuntime(registry=registry, observer=observer)
    return resolved_runtime.run(
        agent=agent,
        session=session,
        prompt=prompt,
        messages=messages,
        tools=tools,
        skills=skills,
        tool_choice=tool_choice,
        tool_execution=tool_execution,
        max_steps=max_steps,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning=reasoning,
        provider_options=provider_options,
        timeout_ms=timeout_ms,
        max_retries=max_retries,
        retry_backoff_ms=retry_backoff_ms,
        stop_on_handoff=stop_on_handoff,
        parent_run_id=parent_run_id,
        idempotency_key=idempotency_key,
    )


def resume_agent(
    *,
    agent: Agent,
    session_id: str,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    tools: ToolSet | ToolRegistry | None = None,
    skills: SkillSet | SkillRegistry | None = None,
    tool_choice: str | ToolChoiceName | None = None,
    tool_execution: ToolExecutionOptions | None = None,
    max_steps: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning: ReasoningConfig | None = None,
    provider_options: dict[str, Any] | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
    stop_on_handoff: bool = False,
    runtime: AgentRuntime | None = None,
    registry: AgentRegistry | None = None,
    observer: AgentObserver | None = None,
) -> Awaitable[AgentRunResult]:
    resolved_runtime = runtime or AgentRuntime(registry=registry, observer=observer)

    async def runner() -> AgentRunResult:
        resumed_session = await load_agent_session(agent, session_id)
        latest_checkpoint: AgentCheckpoint | None = None
        if agent.checkpoint_store is not None:
            latest_checkpoint = await agent.checkpoint_store.get_latest(session_id=session_id)
            if latest_checkpoint is not None:
                resumed_session.metadata = {
                    **resumed_session.metadata,
                    "resumed_from_checkpoint": {
                        "run_id": latest_checkpoint.run_id,
                        "step_index": latest_checkpoint.step_index,
                        "saved_at_ms": latest_checkpoint.saved_at_ms,
                    },
                }
        return await resolved_runtime.run(
            agent=agent,
            session=resumed_session,
            prompt=prompt,
            messages=messages,
            tools=tools,
            skills=skills,
            tool_choice=tool_choice,
            tool_execution=tool_execution,
            max_steps=max_steps,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning,
            provider_options=provider_options,
            timeout_ms=timeout_ms,
            max_retries=max_retries,
            retry_backoff_ms=retry_backoff_ms,
            stop_on_handoff=stop_on_handoff,
            resumed_from_checkpoint=latest_checkpoint,
        )

    return runner()


def stream_agent(
    *,
    agent: Agent,
    session: AgentSession | None = None,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    tools: ToolSet | ToolRegistry | None = None,
    skills: SkillSet | SkillRegistry | None = None,
    tool_choice: str | ToolChoiceName | None = None,
    tool_execution: ToolExecutionOptions | None = None,
    max_steps: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning: ReasoningConfig | None = None,
    provider_options: dict[str, Any] | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
    stop_on_handoff: bool = False,
    runtime: AgentRuntime | None = None,
    registry: AgentRegistry | None = None,
    observer: AgentObserver | None = None,
) -> AgentStreamResult:
    resolved_runtime = runtime or AgentRuntime(registry=registry, observer=observer)
    broadcast = _Broadcast(history=[])

    async def emit(event: AgentEvent) -> None:
        await broadcast.publish(event)

    async def runner() -> AgentRunResult:
        try:
            return await resolved_runtime.run(
                agent=agent,
                session=session,
                prompt=prompt,
                messages=messages,
                tools=tools,
                skills=skills,
                tool_choice=tool_choice,
                tool_execution=tool_execution,
                max_steps=max_steps,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning,
                provider_options=provider_options,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                retry_backoff_ms=retry_backoff_ms,
                stop_on_handoff=stop_on_handoff,
                emit=emit,
                live_stream=True,
            )
        finally:
            await broadcast.close()

    return AgentStreamResult(asyncio.create_task(runner()), broadcast)


def stream_live_agent(
    *,
    agent: Agent,
    session: AgentSession | None = None,
    tools: ToolSet | ToolRegistry | None = None,
    skills: SkillSet | SkillRegistry | None = None,
    tool_choice: str | ToolChoiceName | None = None,
    connect_options: RealtimeConnectOptions | None = None,
    realtime_config: RealtimeSessionConfig | None = None,
    provider_options: dict[str, Any] | None = None,
    runtime: AgentRuntime | None = None,
    registry: AgentRegistry | None = None,
    observer: AgentObserver | None = None,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
) -> LiveAgentStreamResult:
    if not hasattr(agent.model, "connect"):
        raise ValidationError("stream_live_agent() requires an agent.model that supports realtime sessions.")

    resolved_runtime = runtime or AgentRuntime(registry=registry, observer=observer)
    broadcast = _LiveBroadcast(history=[])
    live_session_future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

    async def emit_agent(event: AgentEvent) -> None:
        await broadcast.publish(event)

    async def emit_live(event: RealtimeEvent) -> None:
        await broadcast.publish(event)

    async def runner() -> AgentRunResult:
        resolved_session = session or create_agent_session()
        if agent.memory is not None and not resolved_session.messages and resolved_session.summary is None:
            state = await agent.memory.load(resolved_session.id)
            resolved_session.messages = list(state.messages)
            resolved_session.summary = state.summary
            resolved_session.metadata = {**state.metadata, **resolved_session.metadata}

        run_id = _new_id("run")
        trace = AgentTrace(
            run_id=run_id,
            session_id=resolved_session.id,
            agent_name=agent.name,
            started_at_ms=_now_ms(),
            orchestration_path=[agent.name],
        )
        await emit_agent(AgentRunStartEvent(run_id=run_id, session_id=resolved_session.id, agent_name=agent.name))
        active_skill_activations, skipped_skills, skill_tools = await _select_active_skills(
            _resolve_skill_registry(agent, skills),
            agent=agent,
            session=resolved_session,
            prompt=prompt,
            messages=messages,
        )
        await _emit_skill_events(active_skills=active_skill_activations, skipped_skills=skipped_skills, emit=emit_agent)
        registry_instance = _resolve_tool_registry(agent, tools)
        if skill_tools:
            registry_instance = registry_instance.merge(skill_tools)
        context = AgentContext(
            run_id=run_id,
            session_id=resolved_session.id,
            agent_name=agent.name,
            memory_summary=resolved_session.summary,
            metadata={
                **dict(agent.metadata),
                "skills": [item.skill.name for item in active_skill_activations],
            },
            handoff_path=list(trace.orchestration_path),
        )
        input_messages = _build_run_messages(
            agent=agent,
            session=resolved_session,
            prompt=prompt,
            messages=messages,
            active_skills=[item.skill for item in active_skill_activations],
        )
        _persist_active_skills(resolved_session, active_skill_activations)
        await resolved_runtime._run_input_guardrails(
            agent=agent,
            run_id=run_id,
            session_id=resolved_session.id,
            prompt=prompt,
            messages=input_messages,
            context=context,
            trace=trace,
            emit=emit_agent,
        )
        wrapped_tools = resolved_runtime._wrap_agent_tools(
            agent=agent,
            registry=registry_instance,
            run_id=run_id,
            session_id=resolved_session.id,
            trace=trace,
            started_at_ms=trace.started_at_ms,
            context=context,
            emit=emit_agent,
        )
        instruction_base = realtime_config.instructions if realtime_config is not None and realtime_config.instructions is not None else agent.instructions
        combined_instructions = instruction_base
        if active_skill_activations:
            combined_instructions = "\n\n".join(
                [text for text in [instruction_base, *(_skill_system_message(item.skill) for item in active_skill_activations)] if text]
            )
        live_config = realtime_config or RealtimeSessionConfig(
            instructions=combined_instructions,
            tools=wrapped_tools or None,
            tool_choice=cast(Any, tool_choice),
            provider_options=provider_options,
        )
        if realtime_config is not None and provider_options is not None and realtime_config.provider_options is None:
            live_config = replace(realtime_config, provider_options=provider_options)
        if realtime_config is not None and active_skill_activations:
            live_config = replace(live_config, instructions=combined_instructions)

        transcript = list(resolved_session.messages)
        tool_results: list[ToolExecutionResult] = []
        assistant_buffer: list[str] = []
        last_assistant_text = ""
        live_model = cast(RealtimeModel, agent.model)
        live_session = await live_model.connect(config=live_config, options=connect_options)
        live_session_future.set_result(live_session)
        resolved_session.metadata = {
            **resolved_session.metadata,
            "realtime": {
                "provider": getattr(agent.model, "provider", ""),
                "model_id": getattr(agent.model, "model_id", ""),
            },
        }
        try:
            if messages is not None:
                for message in messages:
                    text = _text_from_message(message)
                    if message.role == "user" and text:
                        await live_session.send_text(text)
            elif prompt is not None:
                await live_session.send_text(prompt)

            async for event in live_session.event_stream():
                await emit_live(event)
                if isinstance(event, RealtimeTextDeltaEvent):
                    assistant_buffer.append(event.text_delta)
                    await emit_agent(AgentTextDeltaEvent(text_delta=event.text_delta))
                    continue
                if isinstance(event, RealtimeTranscriptEvent):
                    if event.role == "user" and event.is_final and event.text:
                        transcript.append(create_text_message("user", event.text))
                    if event.role == "assistant" and event.is_final:
                        text = event.text or "".join(assistant_buffer)
                        if text:
                            last_assistant_text = text
                            transcript.append(create_text_message("assistant", text))
                            assistant_buffer.clear()
                    continue
                if isinstance(event, RealtimeToolCallEvent):
                    await emit_agent(AgentToolCallEvent(tool_call=event.tool_call))
                    definition = wrapped_tools.get(event.tool_call.name) if wrapped_tools else None
                    if definition is None or not is_callable_tool_definition(definition) or definition.execute is None:
                        tool_result = ToolExecutionResult(
                            tool_call_id=event.tool_call.id,
                            tool_name=event.tool_call.name,
                            error=ToolExecutionError(message=f'Unknown realtime tool "{event.tool_call.name}".'),
                            is_error=True,
                        )
                    else:
                        call_context = ToolExecutionContext(
                            tool_name=event.tool_call.name,
                            tool_call_id=event.tool_call.id,
                            run_id=run_id,
                            session_id=resolved_session.id,
                            agent_name=agent.name,
                            memory_summary=resolved_session.summary,
                            permissions=list(definition.permissions),
                            source=definition.source,
                            metadata={**context.metadata, **definition.metadata},
                            handoff_path=list(trace.orchestration_path),
                        )
                        try:
                            output = _invoke_tool_callable(definition.execute, event.tool_call.input, call_context)
                            output = await _maybe_await(output)
                            tool_result = ToolExecutionResult(
                                tool_call_id=event.tool_call.id,
                                tool_name=event.tool_call.name,
                                output=output,
                                is_error=False,
                            )
                        except Exception as error:
                            tool_result = ToolExecutionResult(
                                tool_call_id=event.tool_call.id,
                                tool_name=event.tool_call.name,
                                error=ToolExecutionError(message=str(error)),
                                is_error=True,
                            )
                    tool_results.append(tool_result)
                    transcript.append(ModelMessage(role="tool", parts=[tool_result_part(tool_result)]))
                    await live_session.send_tool_result(tool_result)
                    await emit_agent(AgentToolResultEvent(tool_result=tool_result))
                    await emit_live(RealtimeToolResultEvent(tool_result=tool_result))
                    continue
                if isinstance(event, (RealtimeResponseCompletedEvent, RealtimeSessionEndedEvent)):
                    break
            if assistant_buffer and not last_assistant_text:
                last_assistant_text = "".join(assistant_buffer)
                if last_assistant_text:
                    transcript.append(create_text_message("assistant", last_assistant_text))
            await resolved_runtime._run_output_guardrails(
                agent=agent,
                run_id=run_id,
                session_id=resolved_session.id,
                result=None,
                text=last_assistant_text,
                messages=[create_text_message("assistant", last_assistant_text)] if last_assistant_text else [],
                context=context,
                trace=trace,
                emit=emit_agent,
            )
        except Exception as error:
            if not live_session_future.done():
                live_session_future.set_exception(error)
            await emit_agent(AgentErrorEvent(error=error))
            raise
        finally:
            await live_session.aclose()

        resolved_session.messages = _strip_runtime_system_messages(transcript, agent.instructions)
        if agent.memory is not None and _should_refresh_summary(agent.memory, resolved_session):
            resolved_session.summary = await agent.memory.summarize(
                session_id=resolved_session.id,
                state=AgentMemoryState(
                    messages=list(resolved_session.messages),
                    summary=resolved_session.summary,
                    metadata={**dict(resolved_session.metadata), "state": dict(resolved_session.state)},
                ),
                agent=agent,
            )
            await emit_agent(AgentSummaryUpdateEvent(summary=resolved_session.summary))
        if agent.memory is not None:
            await agent.memory.save(
                resolved_session.id,
                AgentMemoryState(
                    messages=list(resolved_session.messages),
                    summary=resolved_session.summary,
                    metadata={**dict(resolved_session.metadata), "state": dict(resolved_session.state)},
                ),
            )

        await emit_agent(
            AgentFinishEvent(
                run_id=run_id,
                session_id=resolved_session.id,
                text=last_assistant_text,
                finish_reason="stop",
            )
        )
        trace.finished_at_ms = _now_ms()
        result = AgentRunResult(
            run_id=run_id,
            agent_name=agent.name,
            session=resolved_session,
            text=last_assistant_text,
            finish_reason="stop",
            steps=[],
            messages=list(resolved_session.messages),
            tool_results=tool_results,
            trace=trace,
            orchestration_path=list(trace.orchestration_path),
        )
        await broadcast.close()
        return result

    async def managed_runner() -> AgentRunResult:
        try:
            return await runner()
        finally:
            await broadcast.close()

    task = asyncio.create_task(managed_runner())
    return LiveAgentStreamResult(task, broadcast, live_session_future)
