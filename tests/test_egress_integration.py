"""Offline authorization checks at the generation/adapter boundary."""

import pytest

from zhivex_ai import (
    EgressDeniedError,
    create_egress_policy,
    create_mock_language_model,
    generate_text,
    openai_compatible,
    stream_text,
)


class Response:
    status_code = 200

    async def json(self):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    async def iter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}'
        yield ""
        yield "data: [DONE]"
        yield ""


def policy(endpoint="https://allowed.example/v1"):
    return create_egress_policy(allow=[{
        "provider": "*", "model": "*", "region": "*", "endpoint": endpoint,
    }])


async def call(model, streaming, **options):
    if streaming:
        return await stream_text(model=model, prompt="private prompt", **options).collect()
    return await generate_text(model=model, prompt="private prompt", **options)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("declaration", [
    {},
    {"egress": {"endpoint": "https://allowed.example/v1"}},
    {"provider_options": {"base_url": "https://allowed.example/v1"}},
])
async def test_denied_destination_never_reaches_fetch(streaming, declaration):
    calls = []

    async def fetch(url, **kwargs):
        calls.append(url)
        return Response()

    model = openai_compatible(name="acme", base_url="https://denied.example/v1", api_key="test", fetch=fetch).language_model("chat")
    guard = policy()
    with pytest.raises(EgressDeniedError):
        await call(model, streaming, egress_policy=guard, **declaration)
    assert calls == []
    assert guard.records[-1].decision == "denied"
    assert guard.records[-1].endpoint == "https://denied.example/v1"


@pytest.mark.parametrize("streaming", [False, True])
async def test_allowed_destination_matches_transport_and_audit(streaming):
    calls = []

    async def fetch(url, **kwargs):
        calls.append(url)
        return Response()

    model = openai_compatible(name="acme", base_url="https://allowed.example/v1/", api_key="test", fetch=fetch).language_model("chat")
    guard = policy()
    output = await call(model, streaming, egress_policy=guard, egress={"endpoint": "https://allowed.example/v1"})
    assert output.text == "ok"
    assert calls == ["https://allowed.example/v1/chat/completions"]
    assert guard.records[-1].endpoint == "https://allowed.example/v1"


@pytest.mark.parametrize("streaming", [False, True])
async def test_declaration_cannot_supply_unknown_destination(streaming):
    model = create_mock_language_model()
    with pytest.raises(EgressDeniedError):
        await call(model, streaming, egress_policy=policy(), egress={"endpoint": "https://allowed.example/v1"})
    assert model.calls == 0


async def test_prepare_step_replacement_is_authorized():
    async def fetch(url, **kwargs):
        raise AssertionError("Denied model must not send a request")

    model = openai_compatible(name="acme", base_url="https://allowed.example/v1", api_key="test", fetch=fetch).language_model("chat")
    replacement = openai_compatible(name="acme", base_url="https://denied.example/v1", api_key="test", fetch=fetch).language_model("chat")
    with pytest.raises(EgressDeniedError):
        await generate_text(model=model, prompt="private", egress_policy=policy(), prepare_step=lambda _: {"model": replacement})


async def test_no_policy_preserves_existing_behavior():
    async def fetch(url, **kwargs):
        return Response()

    model = openai_compatible(name="acme", base_url="https://other.example/v1", api_key="test", fetch=fetch).language_model("chat")
    assert (await generate_text(model=model, prompt="hello")).text == "ok"
