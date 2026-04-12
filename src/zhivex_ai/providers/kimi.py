from __future__ import annotations

from .._http import Fetcher
from ..types import PortableSupport
from .base import create_provider_bundle
from .openai_compat import create_openai_compatible_provider


def create_kimi(
    *,
    api_key: str | None = None,
    base_url: str = "https://api.moonshot.ai/v1",
    fetch: Fetcher | None = None,
):
    native = create_openai_compatible_provider(
        provider_name="kimi",
        env_var="KIMI_API_KEY",
        api_key=api_key,
        base_url=base_url,
        fetch=fetch,
    )
    return create_provider_bundle(
        name="kimi",
        native=native,
        portable_support=PortableSupport(
            text_generation=True,
            streaming=True,
            structured_output=True,
            tools=True,
            embeddings=True,
            grounding=False,
            retrieval=True,
            transcription=False,
            speech=False,
            portable_badge=False,
            tier="compatibility",
        ),
    )
