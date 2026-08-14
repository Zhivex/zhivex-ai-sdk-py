from __future__ import annotations

import os

from .._http import Fetcher
from ..realtime import RealtimeConnectionFactory
from ..types import AgentCapabilities, PortableSupport
from .base import ProviderBundle, create_provider_bundle
from .openai_compat import create_openai_compatible_provider

VLLM_DEFAULT_BASE_URL = "http://localhost:8000/v1"
VLLM_DEFAULT_API_KEY = "vllm"


def create_vllm(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    fetch: Fetcher | None = None,
    realtime_url: str | None = None,
    realtime_connection_factory: RealtimeConnectionFactory | None = None,
) -> ProviderBundle:
    resolved_key = api_key or os.getenv("VLLM_API_KEY") or VLLM_DEFAULT_API_KEY
    resolved_base_url = (base_url or os.getenv("VLLM_BASE_URL") or VLLM_DEFAULT_BASE_URL).rstrip("/")
    native = create_openai_compatible_provider(
        provider_name="vllm",
        env_var="VLLM_API_KEY",
        api_key=resolved_key,
        base_url=resolved_base_url,
        fetch=fetch,
        supports_transcription=True,
        supports_realtime=True,
        realtime_url=realtime_url,
        realtime_connection_factory=realtime_connection_factory,
    )
    return create_provider_bundle(
        name="vllm",
        native=native,
        agent_capabilities=native.language_model("").capabilities.agent_capabilities or AgentCapabilities(),
        portable_support=PortableSupport(
            text_generation=True,
            streaming=True,
            structured_output=True,
            tools=True,
            embeddings=True,
            grounding=False,
            retrieval=True,
            transcription=True,
            speech=False,
            portable_badge=True,
            tier="portable",
        ),
    )
