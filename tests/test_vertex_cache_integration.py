from argparse import Namespace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("google.auth", reason="Vertex runner tests require the vertex extra")
pytest.importorskip("google.auth.transport.requests", reason="Vertex ADC requires requests")

from scripts import verify_vertex_native_integration as runner
from zhivex_ai.errors import ProviderHTTPError


class VertexCacheIntegrationTests(IsolatedAsyncioTestCase):
    async def test_failed_generation_cleans_only_created_cache_and_requires_absence(self):
        for deleted in (True, False):
            with self.subTest(deleted=deleted), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = Namespace(wheel=root / "wheel", output=root / "report", project="p",
                                 location="global", model="gemini-3.8-flash", cache_only=True)
                name = "projects/p/locations/global/cachedContents/owned"
                cache = AsyncMock()
                cache.create.return_value = SimpleNamespace(name=name)
                cache.get.side_effect = [SimpleNamespace(name=name),
                    ProviderHTTPError("missing", 404) if deleted else SimpleNamespace(name=name)]
                provider = SimpleNamespace(caches=lambda: cache, native=SimpleNamespace(
                    agent_platform=lambda: None, language_model=lambda model: object()))
                with patch.object(runner, "verify_wheel", return_value="digest"), \
                     patch.object(runner.google.auth, "default", return_value=(object(), None)), \
                     patch.object(runner, "create_vertex", return_value=provider), \
                     patch.object(runner, "generate_text", new=AsyncMock(side_effect=RuntimeError("generation failure"))):
                    self.assertEqual(await runner.run(args), 1)
                cache.delete.assert_awaited_once()
                self.assertEqual(cache.delete.call_args.args[0], name)
                report = json.loads(args.output.read_text())
                self.assertEqual(report["operations"]["cache-generation"], "failed")
                self.assertEqual(report["operations"]["cache-absent"], "passed" if deleted else "failed")
