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

    async def aiter_bytes(self):
        raw = self.text.encode("utf-8")
        midpoint = len(raw) // 2
        yield raw[:midpoint]
        yield raw[midpoint:]

    async def aclose(self) -> None:
        self.closed = True

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeBufferedHTTPResponse(FakeStreamingHTTPResponse):
    async def aiter_lines(self):
        raise AssertionError("Buffered requests should not iterate streamed lines.")


class FakeLargeBufferedHTTPResponse(FakeBufferedHTTPResponse):
    async def aread(self) -> bytes:
        raise AssertionError("The bounded path must not call aread().")

    async def aiter_bytes(self):
        yield b"x" * 6
        yield b"x" * 6
        raise AssertionError("Reading must stop as soon as the limit is crossed.")


class FakeAsyncClient:
    last_instance: FakeAsyncClient | None = None
    instances: list[FakeAsyncClient] = []

    def __init__(self, *, timeout: float | None = None) -> None:
        self.timeout = timeout
        self.closed = False
        self.sent_stream: bool | None = None
        self.request: dict[str, object] | None = None
        self.response = FakeStreamingHTTPResponse()
        FakeAsyncClient.last_instance = self
        FakeAsyncClient.instances.append(self)

    def build_request(self, method: str, url: str, *, headers: dict[str, str], json: dict[str, object], timeout: float | None = None):
        self.request = {"method": method, "url": url, "headers": headers, "json": json, "timeout": timeout}
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


class FakeLargeBufferedAsyncClient(FakeAsyncClient):
    def __init__(self, *, timeout: float | None = None) -> None:
        super().__init__(timeout=timeout)
        self.response = FakeLargeBufferedHTTPResponse()


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
            client = FakeAsyncClient.last_instance
            assert client is not None
            self.assertTrue(client.sent_stream)
            self.assertEqual(client.timeout, 1.234)
            self.assertEqual(lines, ['data: {"chunk":1}', "", 'data: {"chunk":2}', ""])
            self.assertTrue(client.response.closed)
            self.assertFalse(client.closed)
        finally:
            await http_module.aclose_default_clients()
            http_module.httpx.AsyncClient = original_client

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
            client = FakeBufferedAsyncClient.last_instance
            assert client is not None
            self.assertTrue(client.sent_stream)
            self.assertEqual(body_text, 'data: {"chunk":1}\n\ndata: {"chunk":2}\n')
            self.assertTrue(client.response.closed)
            self.assertFalse(client.closed)
        finally:
            await http_module.aclose_default_clients()
            http_module.httpx.AsyncClient = original_client

    async def test_default_fetch_rejects_oversized_buffered_response(self) -> None:
        original_client = http_module.httpx.AsyncClient
        original_limit = http_module.DEFAULT_MAX_BUFFERED_RESPONSE_BYTES
        http_module.httpx.AsyncClient = FakeLargeBufferedAsyncClient
        http_module.DEFAULT_MAX_BUFFERED_RESPONSE_BYTES = 10
        try:
            with self.assertRaises(http_module.ResponseTooLargeError):
                await http_module.default_fetch(
                    "https://example.com/buffered",
                    headers={"content-type": "application/json"},
                    json_body={"hello": "world"},
                    timeout_ms=500,
                    stream=False,
                )
            client = FakeLargeBufferedAsyncClient.last_instance
            assert client is not None
            self.assertTrue(client.response.closed)
        finally:
            http_module.DEFAULT_MAX_BUFFERED_RESPONSE_BYTES = original_limit
            await http_module.aclose_default_clients()
            http_module.httpx.AsyncClient = original_client

    async def test_streaming_response_stops_caching_after_cache_limit(self) -> None:
        raw = FakeStreamingHTTPResponse()
        response = http_module.StreamingResponse(response=raw, max_cached_bytes=10)

        lines = [line async for line in response.iter_lines()]

        self.assertEqual(lines, ['data: {"chunk":1}', "", 'data: {"chunk":2}', ""])
        self.assertTrue(raw.closed)
        self.assertEqual(await response.read(), b"")

    async def test_streaming_response_read_enforces_limit_incrementally(self) -> None:
        raw = FakeLargeBufferedHTTPResponse()
        response = http_module.StreamingResponse(response=raw, max_body_bytes=10)

        with self.assertRaises(http_module.ResponseTooLargeError):
            await response.read()

        self.assertTrue(raw.closed)

    async def test_limited_reader_rejects_oversized_content_length_before_reading(self) -> None:
        raw = FakeLargeBufferedHTTPResponse()
        raw.headers = {"content-length": "20"}

        with self.assertRaises(http_module.ResponseTooLargeError):
            await http_module._read_limited_body(raw, max_bytes=10, label="response body")

    async def test_default_fetch_reuses_clients_for_matching_timeouts(self) -> None:
        original_client = http_module.httpx.AsyncClient
        http_module.httpx.AsyncClient = FakeBufferedAsyncClient
        FakeAsyncClient.instances.clear()
        try:
            await http_module.default_fetch(
                "https://example.com/one",
                headers={"content-type": "application/json"},
                json_body={"hello": "world"},
                timeout_ms=500,
                stream=False,
            )
            await http_module.default_fetch(
                "https://example.com/two",
                headers={"content-type": "application/json"},
                json_body={"hello": "again"},
                timeout_ms=500,
                stream=False,
            )
        finally:
            await http_module.aclose_default_clients()
            http_module.httpx.AsyncClient = original_client

        self.assertEqual(len(FakeAsyncClient.instances), 1)
