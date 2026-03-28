from __future__ import annotations

from .._http import Fetcher
from .openai_compat import create_openai_compatible_provider


def create_kimi(
    *,
    api_key: str | None = None,
    base_url: str = "https://api.moonshot.ai/v1",
    fetch: Fetcher | None = None,
):
    return create_openai_compatible_provider(
        provider_name="kimi",
        env_var="KIMI_API_KEY",
        api_key=api_key,
        base_url=base_url,
        fetch=fetch,
    )
