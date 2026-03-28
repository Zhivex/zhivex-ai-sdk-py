from __future__ import annotations

from .._http import Fetcher
from .openai_compat import create_openai_compatible_provider


def create_openrouter(
    *,
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    fetch: Fetcher | None = None,
):
    return create_openai_compatible_provider(
        provider_name="openrouter",
        env_var="OPENROUTER_API_KEY",
        api_key=api_key,
        base_url=base_url,
        fetch=fetch,
    )
