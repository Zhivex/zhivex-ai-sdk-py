from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import os
from typing import Any

from .._http import Fetcher, default_fetch
from ..errors import ConfigurationError
from ..realtime import RealtimeConnectionFactory
from ..types import PortableSupport
from .base import create_provider_bundle
from .openai_compat import (
    OpenAICompatibleBatchesClient,
    OPENAI_COMPAT_CAPABILITIES,
    OpenAICompatibleConversationsClient,
    OpenAICompatibleFileSearchStoresClient,
    OpenAICompatibleFilesClient,
    OpenAICompatibleImagesClient,
    OpenAICompatibleModerationsClient,
    OpenAICompatibleResponsesClient,
    OpenAICompatibleUploadsClient,
    create_openai_compatible_provider,
)


def _copy_list(values: list[Any] | None) -> list[Any] | None:
    return deepcopy(values) if values is not None else None


def _copy_dict(values: dict[str, Any] | None) -> dict[str, Any] | None:
    return deepcopy(values) if values is not None else None


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def openai_hosted_tool(tool_type: str, /, **config: Any) -> dict[str, Any]:
    return _drop_none({"type": tool_type, **deepcopy(config)})


def openai_user_location(
    *,
    city: str | None = None,
    country: str | None = None,
    region: str | None = None,
    timezone: str | None = None,
) -> dict[str, Any]:
    return _drop_none(
        {
            "type": "approximate",
            "city": city,
            "country": country,
            "region": region,
            "timezone": timezone,
        }
    )


def openai_web_search_tool(
    *,
    search_context_size: str | None = None,
    search_content_types: list[str] | None = None,
    user_location: dict[str, Any] | None = None,
    tool_type: str = "web_search",
    **extra: Any,
) -> dict[str, Any]:
    return openai_hosted_tool(
        tool_type,
        search_context_size=search_context_size,
        search_content_types=_copy_list(search_content_types),
        user_location=_copy_dict(user_location),
        **extra,
    )


def openai_file_search_filter(*, key: str, operator: str, value: Any) -> dict[str, Any]:
    return {"key": key, "type": operator, "value": deepcopy(value)}


def openai_file_search_filter_group(operator: str, filters: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": operator, "filters": _copy_list(filters) or []}


def openai_file_search_tool(
    *,
    vector_store_ids: list[str],
    filters: dict[str, Any] | None = None,
    max_num_results: int | None = None,
    ranking_options: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return openai_hosted_tool(
        "file_search",
        vector_store_ids=list(vector_store_ids),
        filters=_copy_dict(filters),
        max_num_results=max_num_results,
        ranking_options=_copy_dict(ranking_options),
        **extra,
    )


def openai_image_mask(*, file_id: str | None = None, image_url: str | None = None) -> dict[str, Any]:
    return _drop_none({"file_id": file_id, "image_url": image_url})


def openai_image_generation_tool(
    *,
    model: str | None = None,
    action: str | None = None,
    background: str | None = None,
    size: str | None = None,
    quality: str | None = None,
    output_format: str | None = None,
    output_compression: int | None = None,
    moderation: str | None = None,
    partial_images: int | None = None,
    input_fidelity: str | None = None,
    input_image_mask: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return openai_hosted_tool(
        "image_generation",
        model=model,
        action=action,
        background=background,
        size=size,
        quality=quality,
        output_format=output_format,
        output_compression=output_compression,
        moderation=moderation,
        partial_images=partial_images,
        input_fidelity=input_fidelity,
        input_image_mask=_copy_dict(input_image_mask),
        **extra,
    )


def openai_network_policy_disabled() -> dict[str, Any]:
    return {"type": "disabled"}


def openai_network_policy_allowlist(
    *,
    allowed_domains: list[str],
    domain_secrets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return _drop_none(
        {
            "type": "allowlist",
            "allowed_domains": list(allowed_domains),
            "domain_secrets": _copy_list(domain_secrets),
        }
    )


def openai_domain_secret(*, domain: str, name: str, value: str) -> dict[str, Any]:
    return {"domain": domain, "name": name, "value": value}


def openai_code_interpreter_container(
    *,
    file_ids: list[str] | None = None,
    memory_limit: str | None = None,
    network_policy: dict[str, Any] | None = None,
    container_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    if container_id is not None:
        return openai_hosted_tool("container_reference", container_id=container_id, **extra)
    return openai_hosted_tool(
        "auto",
        file_ids=_copy_list(file_ids),
        memory_limit=memory_limit,
        network_policy=_copy_dict(network_policy),
        **extra,
    )


def openai_code_interpreter_tool(
    *,
    container: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return openai_hosted_tool("code_interpreter", container=_copy_dict(container), **extra)


def openai_local_skill(*, name: str, path: str, description: str | None = None) -> dict[str, Any]:
    return _drop_none({"name": name, "path": path, "description": description})


def openai_skill_reference(*, skill_id: str, version: str | None = None) -> dict[str, Any]:
    return _drop_none({"type": "skill_reference", "skill_id": skill_id, "version": version})


def openai_inline_skill_source(*, data: str, media_type: str = "application/zip") -> dict[str, Any]:
    return {"type": "base64", "data": data, "media_type": media_type}


def openai_inline_skill(
    *,
    name: str,
    description: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {"type": "inline", "name": name, "description": description, "source": _copy_dict(source) or {}}


def openai_shell_environment(
    *,
    file_ids: list[str] | None = None,
    memory_limit: str | None = None,
    network_policy: dict[str, Any] | None = None,
    container_id: str | None = None,
    local_skills: list[dict[str, Any]] | None = None,
    use_local: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    if use_local:
        return openai_hosted_tool("local", skills=_copy_list(local_skills), **extra)
    if container_id is not None:
        return openai_hosted_tool("container_reference", container_id=container_id, **extra)
    return openai_hosted_tool(
        "container_auto",
        file_ids=_copy_list(file_ids),
        memory_limit=memory_limit,
        network_policy=_copy_dict(network_policy),
        skills=_copy_list(local_skills),
        **extra,
    )


def openai_custom_tool_format_text() -> dict[str, Any]:
    return {"type": "text"}


def openai_custom_tool_format_grammar(*, syntax: str, definition: str) -> dict[str, Any]:
    return {"type": "grammar", "syntax": syntax, "definition": definition}


def openai_custom_tool(
    *,
    name: str,
    description: str | None = None,
    format: dict[str, Any] | None = None,
    defer_loading: bool | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return openai_hosted_tool(
        "custom",
        name=name,
        description=description,
        format=_copy_dict(format),
        defer_loading=defer_loading,
        **extra,
    )


def openai_namespace_tool(
    *,
    name: str,
    description: str,
    tools: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    return openai_hosted_tool(
        "namespace",
        name=name,
        description=description,
        tools=_copy_list(tools),
        **extra,
    )


def openai_tool_search_tool(
    *,
    description: str | None = None,
    execution: str | None = None,
    parameters: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return openai_hosted_tool(
        "tool_search",
        description=description,
        execution=execution,
        parameters=_copy_dict(parameters),
        **extra,
    )


def openai_shell_tool(*, environment: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    return openai_hosted_tool("shell", environment=_copy_dict(environment), **extra)


def openai_local_shell_tool(**extra: Any) -> dict[str, Any]:
    return openai_hosted_tool("local_shell", **extra)


def openai_apply_patch_tool(**extra: Any) -> dict[str, Any]:
    return openai_hosted_tool("apply_patch", **extra)


def openai_computer_use_tool(
    *,
    environment: str | None = None,
    display_width: int | None = None,
    display_height: int | None = None,
    tool_type: str = "computer_use_preview",
    **extra: Any,
) -> dict[str, Any]:
    return openai_hosted_tool(
        tool_type,
        environment=environment,
        display_width=display_width,
        display_height=display_height,
        **extra,
    )


def openai_mcp_tool(
    *,
    server_url: str | None = None,
    server_label: str | None = None,
    headers: dict[str, str] | None = None,
    allowed_tools: list[str] | None = None,
    require_approval: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return openai_hosted_tool(
        "mcp",
        server_url=server_url,
        server_label=server_label,
        headers=_copy_dict(headers),
        allowed_tools=_copy_list(allowed_tools),
        require_approval=require_approval,
        **extra,
    )


def openai_response_options(
    *,
    tools: list[dict[str, Any]] | None = None,
    background: bool | None = None,
    conversation: str | None = None,
    previous_response_id: str | None = None,
    include: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    store: bool | None = None,
    prompt: dict[str, Any] | None = None,
    service_tier: str | None = None,
    truncation: str | dict[str, Any] | None = None,
    user: str | None = None,
    safety_identifier: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return _drop_none(
        {
            "tools": _copy_list(tools),
            "background": background,
            "conversation": conversation,
            "previous_response_id": previous_response_id,
            "include": _copy_list(include),
            "metadata": _copy_dict(metadata),
            "store": store,
            "prompt": _copy_dict(prompt),
            "service_tier": service_tier,
            "truncation": deepcopy(truncation),
            "user": user,
            "safety_identifier": safety_identifier,
            **deepcopy(extra),
        }
    )


def create_openai(
    *,
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    fetch: Fetcher | None = None,
    realtime_url: str | None = None,
    browser_token_url: str | None = None,
    realtime_connection_factory: RealtimeConnectionFactory | None = None,
):
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise ConfigurationError("Missing openai API key.")
    requester = fetch or default_fetch
    base = base_url.rstrip("/")
    native = create_openai_compatible_provider(
        provider_name="openai",
        env_var="OPENAI_API_KEY",
        api_key=resolved_key,
        base_url=base,
        fetch=requester,
        capabilities=replace(OPENAI_COMPAT_CAPABILITIES, files=True),
        supports_audio=True,
        supports_grounding=True,
        supports_realtime=True,
        realtime_url=realtime_url,
        browser_token_url=browser_token_url,
        realtime_connection_factory=realtime_connection_factory,
        files_client_factory=lambda: OpenAICompatibleFilesClient(
            provider="openai",
            api_key=resolved_key,
            base_url=base,
            fetch=requester,
        ),
        images_client_factory=lambda: OpenAICompatibleImagesClient(
            provider="openai",
            api_key=resolved_key,
            base_url=base,
            fetch=requester,
        ),
        uploads_client_factory=lambda: OpenAICompatibleUploadsClient(
            provider="openai",
            api_key=resolved_key,
            base_url=base,
            fetch=requester,
        ),
        moderations_client_factory=lambda: OpenAICompatibleModerationsClient(
            provider="openai",
            api_key=resolved_key,
            base_url=base,
            fetch=requester,
        ),
        batches_client_factory=lambda: OpenAICompatibleBatchesClient(
            provider="openai",
            api_key=resolved_key,
            base_url=base,
            fetch=requester,
        ),
        file_search_stores_client_factory=lambda: OpenAICompatibleFileSearchStoresClient(
            provider="openai",
            api_key=resolved_key,
            base_url=base,
            fetch=requester,
        ),
        responses_client_factory=lambda: OpenAICompatibleResponsesClient(
            provider="openai",
            model_id="",
            api_key=resolved_key,
            base_url=base,
            fetch=requester,
        ),
        conversations_client_factory=lambda: OpenAICompatibleConversationsClient(
            provider="openai",
            model_id="",
            api_key=resolved_key,
            base_url=base,
            fetch=requester,
        ),
    )
    return create_provider_bundle(
        name="openai",
        native=native,
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
