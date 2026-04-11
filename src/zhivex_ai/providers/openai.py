from __future__ import annotations

from .._http import Fetcher
from ..realtime import RealtimeConnectionFactory
from .openai_compat import create_openai_compatible_provider


def create_openai(
    *,
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    fetch: Fetcher | None = None,
    realtime_url: str | None = None,
    browser_token_url: str | None = None,
    realtime_connection_factory: RealtimeConnectionFactory | None = None,
):
    return create_openai_compatible_provider(
        provider_name="openai",
        env_var="OPENAI_API_KEY",
        api_key=api_key,
        base_url=base_url,
        fetch=fetch,
        supports_audio=True,
        supports_grounding=True,
        supports_realtime=True,
        realtime_url=realtime_url,
        browser_token_url=browser_token_url,
        realtime_connection_factory=realtime_connection_factory,
    )
