from __future__ import annotations

import os
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any

from .._http import Fetcher, ResponseLike, default_fetch
from ..errors import ConfigurationError
from ..messages import (
    get_last_provider_data_entry,
    get_provider_data_entries,
    get_provider_data_parts,
    hosted_tool,
    provider_data_part,
)
from ..realtime import RealtimeConnection, RealtimeConnectionFactory, open_websocket_connection
from ..types import (
    AgentCapabilities,
    AzureOpenAIMcpApprovalRequest,
    AzureOpenAIMcpApprovalResponse,
    AzureOpenAIMcpCall,
    AzureOpenAIMcpListTools,
    AzureOpenAIProviderData,
    AzureOpenAIResponseReference,
    HostedToolDefinition,
    PortableSupport,
    ProviderDataPart,
    RealtimeConnectOptions,
    ToolCall,
)
from .base import ProviderBundle, create_provider_bundle
from .openai_compat import (
    OpenAICompatibleConversationsClient,
    OpenAICompatibleFileSearchStoresClient,
    OpenAICompatibleResponsesClient,
    _parse_provider_data_value,
    create_openai_compatible_provider,
)

AzureOpenAITokenProvider = Callable[[], str | Awaitable[str]]
_ENTRA_AUTH_SENTINEL = "__zhivex_azure_entra_token__"


@dataclass(slots=True)
class _AzureOpenAIAuth:
    api_key: str | None = None
    entra_token: str | None = None
    entra_token_provider: AzureOpenAITokenProvider | None = None

    async def _resolved_entra_token(self) -> str:
        if self.entra_token is not None:
            token = self.entra_token
        elif self.entra_token_provider is not None:
            maybe_token = self.entra_token_provider()
            token = await maybe_token if isawaitable(maybe_token) else maybe_token
        else:
            raise ConfigurationError("Missing Azure OpenAI Entra ID token.")
        if not token:
            raise ConfigurationError("Azure OpenAI Entra ID token provider returned an empty token.")
        return token

    async def headers(self, headers: dict[str, str]) -> dict[str, str]:
        resolved = dict(headers)
        for key in list(resolved):
            if key.lower() in {"api-key", "authorization"}:
                resolved.pop(key, None)
        if self.api_key is not None:
            resolved["api-key"] = self.api_key
        else:
            resolved["authorization"] = f"Bearer {await self._resolved_entra_token()}"
        return resolved


@dataclass(slots=True)
class _AzureOpenAIAuthFetcher:
    fetch: Fetcher
    auth: _AzureOpenAIAuth

    async def __call__(
        self,
        url: str,
        *,
        method: str = "POST",
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        body: Any = None,
        timeout_ms: int | None,
        stream: bool = False,
    ) -> ResponseLike:
        return await self.fetch(
            url,
            method=method,
            headers=await self.auth.headers(headers),
            json_body=json_body,
            body=body,
            timeout_ms=timeout_ms,
            stream=stream,
        )


def _azure_openai_realtime_connection_factory(
    *,
    auth: _AzureOpenAIAuth,
    connection_factory: RealtimeConnectionFactory | None,
) -> RealtimeConnectionFactory:
    async def connect(
        url: str,
        headers: dict[str, str],
        options: RealtimeConnectOptions | None,
    ) -> RealtimeConnection:
        resolved_headers = await auth.headers(headers)
        factory = connection_factory or (lambda u, h, o: open_websocket_connection(u, headers=h, options=o))
        return await factory(url, resolved_headers, options)

    return connect


def azure_openai_web_search_tool(
    *,
    search_context_size: str | None = None,
    user_location: dict[str, object] | None = None,
    tool_type: str = "web_search_preview",
    **extra: object,
) -> HostedToolDefinition:
    return hosted_tool(
        name="web_search",
        provider="azure-openai",
        type=tool_type,
        tool_class="web-search",
        config={
            key: value
            for key, value in {
                "search_context_size": search_context_size,
                "user_location": user_location,
                **extra,
            }.items()
            if value is not None
        },
    )


def azure_openai_file_search_tool(
    *,
    vector_store_ids: list[str],
    filters: dict[str, object] | None = None,
    max_num_results: int | None = None,
    **extra: object,
) -> HostedToolDefinition:
    return hosted_tool(
        name="file_search",
        provider="azure-openai",
        type="file_search",
        tool_class="file-search",
        config={
            key: value
            for key, value in {
                "vector_store_ids": list(vector_store_ids),
                "filters": filters,
                "max_num_results": max_num_results,
                **extra,
            }.items()
            if value is not None
        },
    )


def azure_openai_mcp_tool(
    *,
    server_url: str | None = None,
    server_label: str | None = None,
    headers: dict[str, str] | None = None,
    allowed_tools: list[str] | None = None,
    require_approval: str | None = None,
    **extra: object,
) -> HostedToolDefinition:
    return hosted_tool(
        name=server_label or "mcp",
        provider="azure-openai",
        type="mcp",
        tool_class="remote-mcp",
        requires_approval=require_approval != "never",
        config={
            key: value
            for key, value in {
                "server_url": server_url,
                "server_label": server_label,
                "headers": headers,
                "allowed_tools": allowed_tools,
                "require_approval": require_approval,
                **extra,
            }.items()
            if value is not None
        },
    )


def azure_openai_computer_use_tool(
    *,
    environment: str | None = None,
    display_width: int | None = None,
    display_height: int | None = None,
    tool_type: str = "computer_use_preview",
    **extra: object,
) -> HostedToolDefinition:
    return hosted_tool(
        name="computer",
        provider="azure-openai",
        type=tool_type,
        tool_class="computer-use",
        config={
            key: value
            for key, value in {
                "environment": environment,
                "display_width": display_width,
                "display_height": display_height,
                **extra,
            }.items()
            if value is not None
        },
    )


def azure_openai_mcp_approval_response(
    *,
    approval_request_id: str,
    approve: bool,
    id: str | None = None,
    reason: str | None = None,
) -> ProviderDataPart:
    return provider_data_part(
        "azure-openai",
        AzureOpenAIMcpApprovalResponse(
            approval_request_id=approval_request_id,
            approve=approve,
            id=id,
            reason=reason,
        ),
    )


def azure_openai_response_reference(*, response_id: str) -> ProviderDataPart:
    return provider_data_part("azure-openai", AzureOpenAIResponseReference(response_id=response_id))


def parse_azure_openai_provider_data_part(part: ProviderDataPart) -> AzureOpenAIProviderData | None:
    if getattr(part, "type", None) != "provider-data":
        return None
    if getattr(part, "provider", "") not in {"azure-openai", "openai", ""}:
        return None
    data = _parse_provider_data_value(getattr(part, "data", None), "azure-openai")
    if isinstance(
        data,
        (
            AzureOpenAIResponseReference,
            AzureOpenAIMcpApprovalRequest,
            AzureOpenAIMcpApprovalResponse,
            AzureOpenAIMcpCall,
            AzureOpenAIMcpListTools,
        ),
    ):
        return data
    return None


def get_azure_openai_response_reference(value: Any) -> AzureOpenAIResponseReference | None:
    for part in reversed(get_provider_data_parts(value)):
        if getattr(part, "provider", "") not in {"azure-openai", "openai"}:
            continue
        parsed = parse_azure_openai_provider_data_part(part)
        if isinstance(parsed, AzureOpenAIResponseReference):
            return parsed
    return None


def get_azure_openai_response_id(value: Any) -> str | None:
    reference = get_azure_openai_response_reference(value)
    return reference.response_id if reference is not None else None


def get_azure_openai_provider_data(value: Any, *, data_type: str | None = None) -> list[AzureOpenAIProviderData]:
    return get_provider_data_entries(
        value,
        provider="azure-openai",
        parser=parse_azure_openai_provider_data_part,
        data_type=data_type,
    )


def get_last_azure_openai_provider_data(value: Any, *, data_type: str | None = None) -> AzureOpenAIProviderData | None:
    return get_last_provider_data_entry(
        value,
        provider="azure-openai",
        parser=parse_azure_openai_provider_data_part,
        data_type=data_type,
    )


def get_azure_openai_mcp_calls(value: Any) -> list[AzureOpenAIMcpCall]:
    return [
        entry
        for entry in get_azure_openai_provider_data(value, data_type="mcp_call")
        if isinstance(entry, AzureOpenAIMcpCall)
    ]


def get_last_azure_openai_mcp_call(value: Any) -> AzureOpenAIMcpCall | None:
    entry = get_last_azure_openai_provider_data(value, data_type="mcp_call")
    return entry if isinstance(entry, AzureOpenAIMcpCall) else None


def get_azure_openai_mcp_list_tools_events(value: Any) -> list[AzureOpenAIMcpListTools]:
    return [
        entry
        for entry in get_azure_openai_provider_data(value, data_type="mcp_list_tools")
        if isinstance(entry, AzureOpenAIMcpListTools)
    ]


def get_last_azure_openai_mcp_list_tools_event(value: Any) -> AzureOpenAIMcpListTools | None:
    entry = get_last_azure_openai_provider_data(value, data_type="mcp_list_tools")
    return entry if isinstance(entry, AzureOpenAIMcpListTools) else None


def azure_openai_provider_data_tool_call(value: ProviderDataPart | AzureOpenAIProviderData) -> ToolCall | None:
    parsed = parse_azure_openai_provider_data_part(value) if isinstance(value, ProviderDataPart) else value
    if isinstance(parsed, AzureOpenAIMcpCall):
        return ToolCall(
            id=parsed.id,
            name=parsed.name,
            input=_parse_json_string(parsed.arguments),
            provider_metadata={
                key: value
                for key, value in {
                    "provider": "azure-openai",
                    "provider_managed": True,
                    "provider_event_type": parsed.type,
                    "server_label": parsed.server_label,
                    "approval_request_id": parsed.approval_request_id,
                    "status": parsed.status,
                }.items()
                if value is not None
            },
        )
    if isinstance(parsed, AzureOpenAIMcpListTools):
        identifier = parsed.id or f"azure_openai_mcp_list_tools_{parsed.server_label or 'default'}"
        return ToolCall(
            id=identifier,
            name="mcp_list_tools",
            input={key: value for key, value in {"server_label": parsed.server_label, "tools": parsed.tools}.items() if value is not None},
            provider_metadata={
                key: value
                for key, value in {
                    "provider": "azure-openai",
                    "provider_managed": True,
                    "provider_event_type": parsed.type,
                    "server_label": parsed.server_label,
                }.items()
                if value is not None
            },
        )
    return None


def _parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def create_azure_openai(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    api_version: str = "2024-10-21",
    entra_token: str | None = None,
    entra_token_provider: AzureOpenAITokenProvider | None = None,
    fetch: Fetcher | None = None,
    realtime_url: str | None = None,
    browser_token_url: str | None = None,
    realtime_connection_factory: RealtimeConnectionFactory | None = None,
) -> ProviderBundle:
    resolved_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
    resolved_endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    has_entra_auth = entra_token is not None or entra_token_provider is not None
    if resolved_key and has_entra_auth:
        raise ConfigurationError("Configure either Azure OpenAI API key or Entra ID authentication, not both.")
    if not resolved_key and not has_entra_auth:
        raise ConfigurationError("Missing Azure OpenAI credentials. Provide an API key or Entra ID token provider.")
    if not resolved_endpoint:
        raise ConfigurationError("Missing Azure OpenAI endpoint.")
    # Azure OpenAI v1 uses versionless /openai/v1 endpoints and rejects api-version query params.
    base_url = f"{resolved_endpoint.rstrip('/')}/openai/v1"
    auth = _AzureOpenAIAuth(
        api_key=resolved_key,
        entra_token=entra_token,
        entra_token_provider=entra_token_provider,
    )
    requester = _AzureOpenAIAuthFetcher(fetch=fetch or default_fetch, auth=auth)
    auth_value = resolved_key or _ENTRA_AUTH_SENTINEL
    auth_header = "api-key" if resolved_key else "authorization"
    auth_prefix = "" if resolved_key else "Bearer "
    native = create_openai_compatible_provider(
        provider_name="azure-openai",
        env_var="AZURE_OPENAI_API_KEY",
        api_key=auth_value,
        base_url=base_url,
        fetch=requester,
        auth_header=auth_header,
        auth_prefix=auth_prefix,
        supports_audio=True,
        supports_grounding=True,
        supports_realtime=True,
        realtime_url=realtime_url,
        browser_token_url=browser_token_url,
        realtime_connection_factory=_azure_openai_realtime_connection_factory(
            auth=auth,
            connection_factory=realtime_connection_factory,
        ),
        default_grounding_tool={"type": "web_search_preview"},
        file_search_stores_client_factory=lambda: OpenAICompatibleFileSearchStoresClient(
            provider="azure-openai",
            api_key=auth_value,
            base_url=base_url,
            fetch=requester,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
        ),
        responses_client_factory=lambda: OpenAICompatibleResponsesClient(
            provider="azure-openai",
            model_id="",
            api_key=auth_value,
            base_url=base_url,
            fetch=requester,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
        ),
        conversations_client_factory=lambda: OpenAICompatibleConversationsClient(
            provider="azure-openai",
            model_id="",
            api_key=auth_value,
            base_url=base_url,
            fetch=requester,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
        ),
    )
    return create_provider_bundle(
        name="azure-openai",
        native=native,
        agent_capabilities=native.language_model("").capabilities.agent_capabilities or AgentCapabilities(),
        portable_support=PortableSupport(
            text_generation=True,
            streaming=True,
            structured_output=True,
            tools=True,
            embeddings=True,
            grounding=True,
            retrieval=True,
            transcription=True,
            speech=True,
            portable_badge=True,
            tier="portable",
        ),
    )
