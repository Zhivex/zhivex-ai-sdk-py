from __future__ import annotations

from .._http import Fetcher
from .openai_compat import create_openai_compatible_provider


def create_ollama(
    *,
    api_key: str | None = "ollama",
    base_url: str = "http://localhost:11434/v1",
    fetch: Fetcher | None = None,
):
    return create_openai_compatible_provider(
        provider_name="ollama",
        env_var="OLLAMA_API_KEY",
        api_key=api_key,
        base_url=base_url,
        fetch=fetch,
    )
