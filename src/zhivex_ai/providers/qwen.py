from __future__ import annotations

from .._http import Fetcher
from .openai_compat import create_openai_compatible_provider


def create_qwen(
    *,
    api_key: str | None = None,
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    fetch: Fetcher | None = None,
):
    return create_openai_compatible_provider(
        provider_name="qwen",
        env_var="QWEN_API_KEY",
        api_key=api_key,
        base_url=base_url,
        fetch=fetch,
    )
