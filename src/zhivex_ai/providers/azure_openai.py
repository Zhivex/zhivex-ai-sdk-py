from __future__ import annotations

import os

from .._http import Fetcher
from ..errors import ConfigurationError
from .openai_compat import create_openai_compatible_provider


def create_azure_openai(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    api_version: str = "2024-10-21",
    fetch: Fetcher | None = None,
):
    resolved_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
    resolved_endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    if not resolved_key:
        raise ConfigurationError("Missing Azure OpenAI API key.")
    if not resolved_endpoint:
        raise ConfigurationError("Missing Azure OpenAI endpoint.")
    # Azure OpenAI v1 uses versionless /openai/v1 endpoints and rejects api-version query params.
    base_url = f"{resolved_endpoint.rstrip('/')}/openai/v1"
    return create_openai_compatible_provider(
        provider_name="azure-openai",
        env_var="AZURE_OPENAI_API_KEY",
        api_key=resolved_key,
        base_url=base_url,
        fetch=fetch,
        auth_header="api-key",
        auth_prefix="",
        supports_audio=True,
        supports_grounding=True,
    )
