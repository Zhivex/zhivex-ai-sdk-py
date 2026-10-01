"""Regression checks for composed resources and adapter-owned tool constraints."""
from __future__ import annotations

import pytest

from zhivex_ai import AgentCapabilities, ToolChoiceName, ValidationError, UnsupportedFeatureError, hosted_tool
from zhivex_ai.generate_text import _validate_hosted_tools, _validate_tool_choice
from zhivex_ai.providers._native_extensions import NativeExtensions
from zhivex_ai.providers.base import ProviderAdapter
from zhivex_ai.types import ModelCapabilities


class CustomModel:
    provider = "custom-provider"
    model_id = "custom-model"

    def __init__(self, capabilities: AgentCapabilities) -> None:
        self.capabilities = ModelCapabilities(
            streaming=False, tools=True, structured_output=False, json_mode=False,
            tool_choice=True, parallel_tool_calls=False, vision=False, files=False,
            audio_input=False, audio_output=False, embeddings=False, reasoning=False,
            web_search=False, agent_capabilities=capabilities,
        )


def test_custom_provider_declares_named_hosted_tool_choice() -> None:
    model = CustomModel(AgentCapabilities(named_hosted_tool_choice=True, hosted_web_search=True))
    tools = {"search": hosted_tool(name="search", type="web_search", provider=model.provider)}
    _validate_tool_choice(model, tools, ToolChoiceName("search"))
    _validate_hosted_tools(model, tools, ToolChoiceName("search"))
    model.capabilities.agent_capabilities.named_hosted_tool_choice = False
    with pytest.raises(ValidationError, match="provider-managed"):
        _validate_tool_choice(model, tools, ToolChoiceName("search"))


def test_custom_provider_declares_hosted_only_required_constraint() -> None:
    model = CustomModel(AgentCapabilities(hosted_web_search=True, required_hosted_tool_choice=False))
    tools = {"search": hosted_tool(name="search", type="web_search", provider=model.provider)}
    with pytest.raises(UnsupportedFeatureError, match="cannot guarantee"):
        _validate_hosted_tools(model, tools, "required")
    model.capabilities.agent_capabilities.required_hosted_tool_choice = True
    _validate_hosted_tools(model, tools, "required")


def test_custom_provider_declares_hosted_tool_aliases() -> None:
    model = CustomModel(AgentCapabilities(hosted_web_search=True, hosted_tool_provider_aliases=("upstream",)))
    tools = {"search": hosted_tool(name="search", type="web_search", provider="upstream")}
    _validate_hosted_tools(model, tools, "auto")
    model.capabilities.agent_capabilities.hosted_tool_provider_aliases = ()
    with pytest.raises(ValidationError, match="targets provider"):
        _validate_hosted_tools(model, tools, "auto")


def test_native_extensions_are_lazy_isolated_and_retry_failed_initialization() -> None:
    attempts = 0
    client = object()

    def initialize() -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary initialization failure")
        return client

    adapter = ProviderAdapter("custom", lambda _: CustomModel(AgentCapabilities()), live_client_factory=initialize)
    assert attempts == 0
    with pytest.raises(RuntimeError):
        adapter.live()
    assert adapter.live() is client
    assert adapter.live() is client
    assert attempts == 2
    # A provider can compose a new resource without adding adapter fields/methods.
    assert adapter.extensions.resolve("new_resource", lambda: client) is client
    another = NativeExtensions()
    assert another.resolve("new_resource", object) is not client
    assert not hasattr(adapter, "__dict__")


def test_missing_native_resource_preserves_attribute_error() -> None:
    adapter = ProviderAdapter("custom", lambda _: CustomModel(AgentCapabilities()))
    with pytest.raises(AttributeError, match="native voices"):
        adapter.voices()
