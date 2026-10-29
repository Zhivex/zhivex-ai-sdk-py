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
    assert guard.records[-1].endpoint == ("https://denied.example/v1" if declaration else "https://denied.example/v1/chat/completions")


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
    assert guard.records[-1].endpoint == "https://allowed.example/v1/chat/completions"


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


@pytest.mark.parametrize("endpoint", [
    "https://allowed.example/v1/../outside",
    "https://allowed.example/v1/%2e%2e/outside",
    "https://allowed.example/v1/a%2f..%2f..%2foutside",
    "https://allowed.example:invalid/v1",
    "https://[invalid/v1",
])
async def test_ambiguous_or_invalid_endpoint_is_audited_as_denied(endpoint):
    guard = policy()
    with pytest.raises(EgressDeniedError):
        await guard.assert_call({"provider": "acme", "model": "chat", "endpoint": endpoint})
    assert guard.records[-1].decision == "denied"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("model_id", [
    "../../outside", "%2e%2e/%2e%2e/outside", "..%2f..%2foutside",
    "%252e%252e/%252e%252e/outside",
])
async def test_gemini_effective_url_traversal_is_denied(streaming, model_id, native):
    from zhivex_ai import create_gemini
    import httpx

    requests = []

    async def fetch(url, **kwargs):
        requests.append(httpx.Request("POST", url).url)
        return GeminiResponse()

    provider = create_gemini(api_key="private-key", base_url="https://allowed.example/v1", fetch=fetch)
    model = (provider.native if native else provider).language_model(model_id)
    guard = policy()
    with pytest.raises(EgressDeniedError):
        await call(model, streaming, egress_policy=guard)
    assert requests == []
    assert guard.records[-1].decision == "denied"
    assert "private-key" not in repr(guard.records)


class GeminiResponse:
    status_code = 200

    async def json(self):
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}

    async def iter_lines(self):
        import json
        yield "data: " + json.dumps(await self.json())
        yield ""


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("suffix", [";other", "%3Bother", "%3bother"])
async def test_compatible_semicolon_path_cannot_impersonate_allowed_prefix(streaming, suffix):
    import httpx

    requests = []

    async def fetch(url, **kwargs):
        requests.append(httpx.Request("POST", url).url)
        return Response()

    model = openai_compatible(name="acme", base_url="https://allowed.example/v1" + suffix, api_key="test", fetch=fetch).language_model("chat")
    with pytest.raises(EgressDeniedError):
        await call(model, streaming, egress_policy=policy())
    assert requests == []


class OpenAIResponse:
    status_code = 200

    async def json(self):
        return {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}]}

    async def iter_lines(self):
        yield 'data: {"type":"response.output_text.delta","delta":"ok"}'
        yield ""
        yield 'data: {"type":"response.completed","response":{"status":"completed"}}'
        yield ""


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("provider_name", ["gemini", "openai"])
@pytest.mark.parametrize("allowed", [False, True])
async def test_standard_provider_wrapper_uses_effective_destination(streaming, native, provider_name, allowed):
    import httpx
    from zhivex_ai import create_gemini, create_openai

    factory = create_gemini if provider_name == "gemini" else create_openai
    model_id = "gemini-3.8-flash" if provider_name == "gemini" else "gpt-4o-mini"
    requests = []

    async def fetch(url, **kwargs):
        requests.append(httpx.Request("POST", url).url)
        return GeminiResponse() if provider_name == "gemini" else OpenAIResponse()

    provider = factory(api_key="private-key", base_url="https://allowed.example/v1", fetch=fetch)
    model = (provider.native if native else provider).language_model(model_id)
    action = "streamGenerateContent" if streaming else "generateContent"
    path = f"/v1/models/{model_id}:{action}" if provider_name == "gemini" else "/v1/responses"
    # A route-specific rule proves authorization uses the entire request URL.
    guard = policy("https://allowed.example" + (path if allowed else "/v1/other"))
    if allowed:
        output = await call(model, streaming, egress_policy=guard, egress={"endpoint": "https://allowed.example/v1"})
        assert output.text == "ok"
        assert [url.path for url in requests] == [path]
        assert guard.records[-1].endpoint == "https://allowed.example" + path
    else:
        with pytest.raises(EgressDeniedError):
            await call(model, streaming, egress_policy=guard)
        assert requests == []
    assert "private-key" not in repr(guard.records)


@pytest.mark.parametrize("streaming", [False, True])
async def test_explicit_semicolon_path_rule_is_preserved(streaming):
    requests = []

    async def fetch(url, **kwargs):
        requests.append(url)
        return Response()

    base = "https://allowed.example/v1;other"
    model = openai_compatible(name="acme", base_url=base, api_key="test", fetch=fetch).language_model("chat")
    guard = policy(base)
    assert (await call(model, streaming, egress_policy=guard)).text == "ok"
    assert requests == [base + "/chat/completions"]
    assert guard.records[-1].endpoint == requests[0]


@pytest.mark.parametrize("rule, endpoint", [
    ("https://EXAMPLE.test:443/v1", "https://example.test/v1/chat"),
    ("https://[::1]:443/v1", "https://[::1]/v1/chat"),
    ("https://例え.test/v1", "https://xn--r8jz45g.test/v1/chat"),
    ("https://example.test/caf%C3%A9", "https://example.test/café/chat"),
])
async def test_policy_canonicalization_matches_httpx_request(rule, endpoint):
    import httpx

    guard = policy(rule)
    record = await guard.assert_call({"provider": "acme", "model": "chat", "endpoint": endpoint})
    assert record.endpoint == str(httpx.Request("POST", endpoint).url)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("native", [False, True])
async def test_vertex_inherited_route_is_checked(streaming, native):
    from zhivex_ai import create_vertex

    async def fetch(url, **kwargs):
        raise AssertionError("Traversal must be denied before fetch")

    provider = create_vertex(api_key="test", express_mode=True, base_url="https://allowed.example/v1", fetch=fetch)
    model = (provider.native if native else provider).language_model("../../../../outside")
    with pytest.raises(EgressDeniedError):
        await call(model, streaming, egress_policy=policy())
