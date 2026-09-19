from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("google.auth", reason="Vertex runner tests require the vertex extra")
pytest.importorskip("google.auth.transport.requests", reason="Vertex ADC requires requests")

from scripts.verify_vertex_model_discovery import inventory


class VertexModelDiscoveryTests(IsolatedAsyncioTestCase):
    async def test_follows_pages_and_deduplicates_public_model_metadata(self):
        first = {"name": "publishers/google/models/a", "versionId": "1", "unrelated": "omit"}
        second = {"name": "publishers/google/models/b"}
        client = AsyncMock()
        client.list_models.side_effect = [
            {"publisherModels": [first], "nextPageToken": "cursor"},
            {"publisherModels": [first, second]},
        ]
        result = await inventory(client, "google")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["pages"], 2)
        self.assertEqual(result["models"], [
            {"name": first["name"], "versionId": "1"}, second,
        ])
        self.assertEqual(client.list_models.call_args_list[1].kwargs["page_token"], "cursor")

    async def test_repeated_cursor_and_bad_identity_cannot_pass(self):
        client = AsyncMock()
        client.list_models.return_value = {"nextPageToken": "repeat"}
        with self.assertRaisesRegex(ValueError, "repeated"):
            await inventory(client, "google")
        self.assertEqual(client.list_models.call_count, 2)
        client.list_models.return_value = {"publisherModels": [{"name": "publishers/other/models/a"}]}
        with self.assertRaisesRegex(ValueError, "identity"):
            await inventory(client, "google")

    async def test_page_limit_is_incomplete_not_success(self):
        client = AsyncMock()
        client.list_models.return_value = {"publisherModels": [], "nextPageToken": "more"}
        result = await inventory(client, "google", max_pages=1)
        self.assertEqual(result["status"], "incomplete-page-limit")
