from __future__ import annotations

from .._http import Fetcher
from .openai_compat import create_openai_compatible_provider


def create_openai(*, api_key: str | None = None, base_url: str = "https://api.openai.com/v1", fetch: Fetcher | None = None):
    return create_openai_compatible_provider(
        provider_name="openai",
        env_var="OPENAI_API_KEY",
        api_key=api_key,
        base_url=base_url,
        fetch=fetch,
        supports_audio=True,
        supports_grounding=True,
    )
