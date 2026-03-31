from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
HTTP_PATH = ROOT / "src" / "zhivex_ai" / "_http.py"
fake_httpx = types.ModuleType("httpx")


class _FakeHTTPXResponse:
    def __init__(self, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.text = text

    def json(self):
        import json

        return json.loads(self.text)


fake_httpx.Response = _FakeHTTPXResponse
fake_httpx.AsyncClient = object
sys.modules.setdefault("httpx", fake_httpx)
SPEC = importlib.util.spec_from_file_location("zhivex_ai__http_test", HTTP_PATH)
assert SPEC is not None
assert SPEC.loader is not None
http_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = http_module
SPEC.loader.exec_module(http_module)


class FakeStreamingHTTPResponse:
    def __init__(self) -> None:
        self.status_code = 200
        self.closed = False
        self._lines = ['data: {"chunk":1}', "", 'data: {"chunk":2}', ""]
        self.text = "\n".join(self._lines)

    async def aread(self) -> bytes:
        return self.text.encode("utf-8")

    async def aclose(self) -> None:
        self.closed = True

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeBufferedHTTPResponse(FakeStreamingHTTPResponse):
    async def aiter_lines(self):
        raise AssertionError("Buffered requests should not iterate streamed lines.")


class FakeAsyncClient:
    last_instance: FakeAsyncClient | None = None

    def __init__(self, *, timeout: float | None = None) -> None:
        self.timeout = timeout
        self.closed = False
        self.sent_stream: bool | None = None
        self.request: dict[str, object] | None = None
        self.response = FakeStreamingHTTPResponse()
        FakeAsyncClient.last_instance = self

    def build_request(self, method: str, url: str, *, headers: dict[str, str], json: dict[str, object]):
        self.request = {"method": method, "url": url, "headers": headers, "json": json}
        return self.request

    async def send(self, request, *, stream: bool = False):
        self.sent_stream = stream
        return self.response

    async def aclose(self) -> None:
        self.closed = True


class FakeBufferedAsyncClient(FakeAsyncClient):
    def __init__(self, *, timeout: float | None = None) -> None:
        super().__init__(timeout=timeout)
        self.response = FakeBufferedHTTPResponse()


class HttpTests(IsolatedAsyncioTestCase):
    async def test_default_fetch_uses_true_streaming_mode(self) -> None:
        original_client = http_module.httpx.AsyncClient
        http_module.httpx.AsyncClient = FakeAsyncClient
        try:
            response = await http_module.default_fetch(
                "https://example.com/stream",
                headers={"content-type": "application/json"},
                json_body={"hello": "world"},
                timeout_ms=1234,
                stream=True,
            )
            lines = [line async for line in response.iter_lines()]
        finally:
            http_module.httpx.AsyncClient = original_client

        client = FakeAsyncClient.last_instance
        assert client is not None
        self.assertTrue(client.sent_stream)
        self.assertEqual(client.timeout, 1.234)
        self.assertEqual(lines, ['data: {"chunk":1}', "", 'data: {"chunk":2}', ""])
        self.assertTrue(client.response.closed)
        self.assertTrue(client.closed)

    async def test_default_fetch_buffers_non_streaming_requests_and_closes_resources(self) -> None:
        original_client = http_module.httpx.AsyncClient
        http_module.httpx.AsyncClient = FakeBufferedAsyncClient
        try:
            response = await http_module.default_fetch(
                "https://example.com/buffered",
                headers={"content-type": "application/json"},
                json_body={"hello": "world"},
                timeout_ms=500,
                stream=False,
            )
            body_text = await response.text()
        finally:
            http_module.httpx.AsyncClient = original_client

        client = FakeBufferedAsyncClient.last_instance
        assert client is not None
        self.assertFalse(client.sent_stream)
        self.assertEqual(body_text, 'data: {"chunk":1}\n\ndata: {"chunk":2}\n')
        self.assertTrue(client.response.closed)
        self.assertTrue(client.closed)
