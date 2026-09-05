from __future__ import annotations

import asyncio
from unittest import IsolatedAsyncioTestCase

import httpx

from zhivex_ai import ConfigurationError, HTTPTransport, aclose_default_clients
from zhivex_ai import _http


class HTTPLifecycleTests(IsolatedAsyncioTestCase):
    async def test_one_default_pool_per_loop_independent_of_timeout(self):
        try:
            clients = [_http._shared_client(timeout) for timeout in range(50)]
            self.assertTrue(all(client is clients[0] for client in clients))
            first = clients[0]

            async def other_loop():
                second = _http._shared_client(1)
                try:
                    self.assertIsNot(second, first)
                finally:
                    await aclose_default_clients()
                self.assertTrue(second.is_closed)

            await asyncio.to_thread(asyncio.run, other_loop())
            self.assertFalse(first.is_closed)
        finally:
            await aclose_default_clients()
        self.assertTrue(first.is_closed)

    async def test_owned_client_closes_and_cannot_be_reopened(self):
        transport = HTTPTransport()
        async with transport:
            client = transport._get_client()
        self.assertTrue(client.is_closed)
        await transport.aclose()
        with self.assertRaises(ConfigurationError):
            await transport("https://example.test", headers={}, timeout_ms=None)

    async def test_borrowed_client_stays_open_and_timeout_is_per_request(self):
        seen = []

        async def handler(request):
            seen.append(request.extensions["timeout"])
            return httpx.Response(200, json={"ok": True})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async with HTTPTransport(client=client) as transport:
                for timeout_ms in (100, 250, None):
                    response = await transport(
                        "https://example.test", headers={}, timeout_ms=timeout_ms
                    )
                    self.assertEqual(await response.json(), {"ok": True})
            self.assertFalse(client.is_closed)
        self.assertEqual([value["read"] for value in seen], [0.1, 0.25, 300])

    async def test_owned_transport_rejects_cross_loop_use_and_close(self):
        async with HTTPTransport() as transport:

            async def wrong_loop():
                with self.assertRaisesRegex(ConfigurationError, "event loops"):
                    await transport("https://example.test", headers={}, timeout_ms=None)
                with self.assertRaisesRegex(ConfigurationError, "event loops"):
                    await transport.aclose()

            await asyncio.to_thread(asyncio.run, wrong_loop())
            self.assertFalse(transport._get_client().is_closed)
