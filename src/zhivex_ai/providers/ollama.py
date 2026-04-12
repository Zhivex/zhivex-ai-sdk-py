from __future__ import annotations

from .._http import Fetcher
from ..types import PortableSupport
from .base import create_provider_bundle
from .openai_compat import create_openai_compatible_provider


def create_ollama(
    *,
    api_key: str | None = "ollama",
    base_url: str = "http://localhost:11434/v1",
    fetch: Fetcher | None = None,
):
    native = create_openai_compatible_provider(
        provider_name="ollama",
        env_var="OLLAMA_API_KEY",
        api_key=api_key,
        base_url=base_url,
        fetch=fetch,
    )
    return create_provider_bundle(
        name="ollama",
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
