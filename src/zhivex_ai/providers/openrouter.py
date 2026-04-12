from __future__ import annotations

from .._http import Fetcher
from ..types import PortableSupport
from .base import create_provider_bundle
from .openai_compat import create_openai_compatible_provider


def create_openrouter(
    *,
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    fetch: Fetcher | None = None,
):
    native = create_openai_compatible_provider(
        provider_name="openrouter",
        env_var="OPENROUTER_API_KEY",
        api_key=api_key,
        base_url=base_url,
        fetch=fetch,
        supports_speech=True,
        speech_transport="chat_audio",
    )
    return create_provider_bundle(
        name="openrouter",
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
            speech=True,
            portable_badge=False,
            tier="native-only",
        ),
    )
