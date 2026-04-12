from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import create_azure_openai


class AzureOpenAIProviderTests(TestCase):
    def test_azure_openai_uses_versionless_v1_base_url(self) -> None:
        provider = create_azure_openai(
            api_key="test",
            endpoint="https://example.openai.azure.com",
            api_version="2024-10-21",
        )

        model = provider.native.language_model("gpt-4o-mini")
        self.assertEqual(model.base_url, "https://example.openai.azure.com/openai/v1")
