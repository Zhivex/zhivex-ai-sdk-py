"""Local, HTTP and MCP tool execution and discovery adapters."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Iterable
from dataclasses import asdict, fields, is_dataclass
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Literal, cast

from ._agent_contracts import ToolRuntime
from ._http import default_fetch
from ._serde import (
    serialize_mcp_server_config,
    serialize_tool_definition,
    serialize_tool_execution_context,
)
from .errors import ProviderHTTPError, ValidationError
from .messages import is_callable_tool_definition
from .types import (
    AnyToolDefinition,
    MCPServerConfig,
    MCPToolConfig,
    RemoteHTTPToolConfig,
    ToolDefinition,
    ToolExecutionContext,
    ToolSet,
)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _invoke_tool_callable(
    execute: Any, parsed: Any, context: ToolExecutionContext
) -> Any:
    mode = _tool_callable_mode(execute)
    if mode == "kwargs":
        return execute(parsed, context=context)
    if mode == "positional":
        return execute(parsed, context)
    return execute(parsed)


def _stable_fingerprint_value(value: Any, _seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    seen = _seen if _seen is not None else set()
    marker = id(value)
    if marker in seen:
        return {"cycle": f"{value.__class__.__module__}.{value.__class__.__qualname__}"}
    seen.add(marker)
    if isinstance(value, dict):
        return {
            str(key): _stable_fingerprint_value(item, seen)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_fingerprint_value(item, seen) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_stable_fingerprint_value(item, seen) for item in value]
        return sorted(
            normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str)
        )
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _stable_fingerprint_value(getattr(value, item.name), seen)
            for item in fields(value)
            if not item.name.startswith("_")
        }
    state = getattr(value, "__dict__", None)
    if isinstance(state, dict):
        public_state = {
            key: item for key, item in state.items() if not str(key).startswith("_")
        }
        return {
            "type": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "state": _stable_fingerprint_value(public_state, seen),
        }
    return {"type": f"{value.__class__.__module__}.{value.__class__.__qualname__}"}


def _callable_fingerprint(execute: Any) -> dict[str, Any] | None:
    if execute is None:
        return None
    if isinstance(execute, partial):
        return {
            "partial": _callable_fingerprint(execute.func),
            "args": _stable_fingerprint_value(execute.args),
            "keywords": _stable_fingerprint_value(execute.keywords or {}),
        }
    target = inspect.unwrap(execute)
    code = getattr(target, "__code__", None)
    code_digest = ""
    if code is not None:
        material = b"\0".join(
            (
                code.co_code,
                repr(code.co_consts).encode("utf-8", "backslashreplace"),
                repr(code.co_names).encode("utf-8", "backslashreplace"),
                repr(getattr(target, "__defaults__", None)).encode(
                    "utf-8", "backslashreplace"
                ),
                repr(getattr(target, "__kwdefaults__", None)).encode(
                    "utf-8", "backslashreplace"
                ),
            )
        )
        code_digest = hashlib.sha256(material).hexdigest()
    closure: list[Any] = []
    for cell in getattr(target, "__closure__", None) or ():
        try:
            closure.append(_stable_fingerprint_value(cell.cell_contents))
        except ValueError:
            closure.append({"unavailable": True})
    bound_self = getattr(target, "__self__", None)
    return {
        "module": str(getattr(target, "__module__", target.__class__.__module__)),
        "qualname": str(getattr(target, "__qualname__", target.__class__.__qualname__)),
        "code_sha256": code_digest,
        "closure": closure,
        "bound_state": _stable_fingerprint_value(bound_self)
        if bound_self is not None
        else None,
        "callable_state": _stable_fingerprint_value(target)
        if code is None
        and not inspect.ismethod(target)
        and not inspect.isfunction(target)
        else None,
    }


def _tool_definition_fingerprint(definition: ToolDefinition) -> str:
    payload = serialize_tool_definition(definition, redact_credentials=True)
    payload["executor"] = _callable_fingerprint(definition.execute)
    payload["input_guardrails"] = [
        _callable_fingerprint(guardrail) for guardrail in definition.input_guardrails
    ]
    payload["output_guardrails"] = [
        _callable_fingerprint(guardrail) for guardrail in definition.output_guardrails
    ]
    encoded = json.dumps(
        payload, separators=(",", ":"), sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2:
        return "positional"
    return "single"


class LocalToolRuntime:
    async def execute(
        self, definition: AnyToolDefinition, input: Any, context: ToolExecutionContext
    ) -> Any:
        if not is_callable_tool_definition(definition):
            raise RuntimeError(
                f'Tool "{definition.name}" is provider-managed and cannot run in the local tool runtime.'
            )
        if definition.execute is None:
            raise RuntimeError(
                f'Tool "{definition.name}" does not define a local executor.'
            )
        is_async = inspect.iscoroutinefunction(
            definition.execute
        ) or inspect.iscoroutinefunction(getattr(definition.execute, "__call__", None))
        if is_async:
            result = _invoke_tool_callable(definition.execute, input, context)
        else:
            result = await asyncio.to_thread(
                _invoke_tool_callable, definition.execute, input, context
            )
        return await _maybe_await(result)

    async def aclose(self) -> None:
        return None


class UnsupportedToolRuntime:
    def __init__(self, source: str) -> None:
        self._source = source

    async def execute(
        self, definition: AnyToolDefinition, input: Any, context: ToolExecutionContext
    ) -> Any:
        raise RuntimeError(
            f'Tool "{definition.name}" uses source "{self._source}", but no runtime is configured for that source.'
        )

    async def aclose(self) -> None:
        return None


class HTTPRemoteToolRuntime:
    def __init__(self, *, fetch: Any = None) -> None:
        self._fetch = fetch or default_fetch

    async def execute(
        self, definition: AnyToolDefinition, input: Any, context: ToolExecutionContext
    ) -> Any:
        if not is_callable_tool_definition(definition):
            raise RuntimeError(
                f'Tool "{definition.name}" is provider-managed and cannot run in the remote tool runtime.'
            )
        config: RemoteHTTPToolConfig | None = definition.remote_config
        if config is None:
            raise RuntimeError(
                f'Tool "{definition.name}" does not define a remote_config.'
            )
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
            raise RuntimeError(
                f'Remote tool "{definition.name}" returned a non-object payload.'
            )
        if isinstance(payload.get("error"), dict):
            raise RuntimeError(
                str(
                    payload["error"].get("message")
                    or f'Remote tool "{definition.name}" failed.'
                )
            )
        if "output" not in payload:
            raise RuntimeError(
                f'Remote tool "{definition.name}" response must include an "output" field.'
            )
        return payload["output"]

    async def aclose(self) -> None:
        return None


def _normalize_mcp_content_item(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_mcp_content_item(item) for key, item in value.items()
        }
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


def _mcp_result_is_error(payload: Any) -> bool:
    if isinstance(payload, dict):
        value = payload.get("isError", payload.get("is_error"))
    else:
        value = getattr(payload, "isError", getattr(payload, "is_error", None))
    return value is True


class MCPToolRuntime:
    def __init__(self) -> None:
        self._sessions: dict[str, tuple[Any, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _key(self, server: MCPServerConfig) -> str:
        return _json_dumps(serialize_mcp_server_config(server))

    async def _load_client_api(self) -> tuple[Any, Any, Any]:
        try:
            from mcp import ClientSession  # type: ignore[import-not-found]
            from mcp.client.stdio import (  # type: ignore[import-not-found]
                StdioServerParameters,
                stdio_client,
            )
            from mcp.client.streamable_http import (
                streamable_http_client,  # type: ignore[import-not-found]
            )
        except Exception as error:
            raise RuntimeError(
                'MCP support requires the optional dependency "mcp".'
            ) from error
        return (
            ClientSession,
            (StdioServerParameters, stdio_client),
            streamable_http_client,
        )

    async def _get_session(self, server: MCPServerConfig) -> Any:
        key = self._key(server)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._sessions:
                return self._sessions[key][1]
            (
                ClientSession,
                stdio_bundle,
                streamable_http_client,
            ) = await self._load_client_api()
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
                    timeout=server.timeout_ms / 1000
                    if server.timeout_ms is not None
                    else None,
                )
            else:
                raise ValidationError(
                    f'Unsupported MCP transport "{server.transport}".'
                )

            transport = await transport_cm.__aenter__()
            if not isinstance(transport, tuple) or len(transport) != 2:
                raise RuntimeError(
                    "MCP transport client did not return the expected read/write streams."
                )
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

    async def execute(
        self, definition: AnyToolDefinition, input: Any, context: ToolExecutionContext
    ) -> Any:
        if not is_callable_tool_definition(definition):
            raise RuntimeError(
                f'Tool "{definition.name}" is provider-managed and cannot run in the MCP tool runtime.'
            )
        config: MCPToolConfig | None = definition.mcp_config
        if config is None:
            raise RuntimeError(
                f'Tool "{definition.name}" does not define an mcp_config.'
            )
        session = await self._get_session(config.server)
        result = await session.call_tool(config.tool_name, arguments=input)
        if _mcp_result_is_error(result):
            detail = str(_normalize_mcp_result(result))
            if len(detail) > 1000:
                detail = f"{detail[:997]}..."
            raise RuntimeError(
                f'MCP tool "{config.tool_name}" returned an error: {detail}'
            )
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

    async def __aexit__(
        self, exc_type: Any, exc: BaseException | None, tb: Any
    ) -> None:
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

    async def execute(
        self, definition: AnyToolDefinition, input: Any, context: ToolExecutionContext
    ) -> Any:
        if not is_callable_tool_definition(definition):
            raise RuntimeError(
                f'Tool "{definition.name}" is provider-managed and cannot run in the local agent runtime.'
            )
        runtime = self._runtimes.get(
            definition.source, UnsupportedToolRuntime(definition.source)
        )
        return await runtime.execute(definition, input, context)

    async def aclose(self) -> None:
        seen: set[int] = set()
        for runtime in self._runtimes.values():
            marker = id(runtime)
            if marker in seen:
                continue
            seen.add(marker)
            await runtime.aclose()


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
    base_name = (
        _sanitize_tool_name(tool_name)
        if name_transform == "snake_case"
        else str(tool_name)
    )
    resolved_prefix = (
        _sanitize_tool_name(prefix)
        if prefix and name_transform == "snake_case"
        else prefix
    )
    if not resolved_prefix:
        return base_name
    if resolved_prefix.endswith("_"):
        return f"{resolved_prefix}{base_name}"
    return f"{resolved_prefix}_{base_name}"


def _mcp_tool_annotations(item: Any) -> dict[str, bool]:
    raw = (
        item.get("annotations")
        if isinstance(item, dict)
        else getattr(item, "annotations", None)
    )
    names = {
        "read_only": ("readOnlyHint", "read_only_hint"),
        "destructive": ("destructiveHint", "destructive_hint"),
        "idempotent": ("idempotentHint", "idempotent_hint"),
        "open_world": ("openWorldHint", "open_world_hint"),
    }
    normalized: dict[str, bool] = {}
    for target, candidates in names.items():
        for candidate in candidates:
            if isinstance(raw, dict) and candidate in raw:
                value = raw[candidate]
            elif raw is not None and hasattr(raw, candidate):
                value = getattr(raw, candidate)
            else:
                continue
            if isinstance(value, bool):
                normalized[target] = value
            break
    return normalized


def _mcp_tool_security_classification(
    annotations: dict[str, bool],
    *,
    trusted_by_application: bool = False,
) -> tuple[bool, list[str]]:
    # MCP annotations are untrusted hints. They help describe permissions but
    # never grant automatic execution; only the application's exact-name
    # allowlist can opt a discovered tool out of approval.
    if annotations.get("read_only") is True and annotations.get("destructive") is False:
        permissions = ["read", "network"]
    else:
        permissions = ["network", "external-side-effect"]
        if annotations.get("destructive") is True:
            permissions.extend(["write", "delete"])
    return not trusted_by_application, permissions


async def _load_mcp_tool_definitions(
    server: MCPServerConfig,
    *,
    prefix: str | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    trusted_tools: Iterable[str] | None = None,
    name_transform: Literal["preserve", "snake_case"] = "preserve",
) -> ToolSet:
    runtime = MCPToolRuntime()
    try:
        include_set = set(include or [])
        exclude_set = set(exclude or [])
        trusted_tool_names = set(trusted_tools or [])
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
            local_name = _build_mcp_local_tool_name(
                remote_name, prefix=prefix, name_transform=name_transform
            )
            previous_remote_name = seen_remote_names.get(local_name)
            if previous_remote_name is not None and previous_remote_name != remote_name:
                raise ValidationError(
                    f'MCP tool name collision for "{local_name}": "{previous_remote_name}" and "{remote_name}". '
                    "Use a prefix or preserve the original names to disambiguate them."
                )
            seen_remote_names[local_name] = remote_name
            annotations = _mcp_tool_annotations(item)
            trusted_by_application = remote_name in trusted_tool_names
            requires_approval, permissions = _mcp_tool_security_classification(
                annotations,
                trusted_by_application=trusted_by_application,
            )
            tools[local_name] = ToolDefinition(
                name=local_name,
                description=description,
                schema=schema or {},
                execute=None,
                source="mcp",
                requires_approval=requires_approval,
                permissions=permissions,
                metadata={
                    "mcp_server": server.name,
                    "mcp_tool_name": remote_name,
                    "mcp_annotations": annotations,
                    "mcp_trust": "application"
                    if trusted_by_application
                    else "approval-required",
                },
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
    trusted_tools: Iterable[str] | None = None,
) -> ToolSet:
    return await _load_mcp_tool_definitions(
        server,
        prefix=prefix,
        include=include,
        exclude=exclude,
        trusted_tools=trusted_tools,
        name_transform="preserve",
    )


async def create_mcp_tool_registry(
    server: MCPServerConfig,
    *,
    prefix: str | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    trusted_tools: Iterable[str] | None = None,
    name_transform: Literal["preserve", "snake_case"] = "snake_case",
) -> ToolRegistry:
    resolved_prefix = (
        server.name if prefix is None and name_transform == "snake_case" else prefix
    )
    tools = await _load_mcp_tool_definitions(
        server,
        prefix=resolved_prefix,
        include=include,
        exclude=exclude,
        trusted_tools=trusted_tools,
        name_transform=name_transform,
    )
    return ToolRegistry(
        tools,
        runtimes={"mcp": MCPToolRuntime()},
    )


LocalToolRuntime.__module__ = "zhivex_ai.agent"
UnsupportedToolRuntime.__module__ = "zhivex_ai.agent"
HTTPRemoteToolRuntime.__module__ = "zhivex_ai.agent"
MCPToolRuntime.__module__ = "zhivex_ai.agent"
ToolRegistry.__module__ = "zhivex_ai.agent"
