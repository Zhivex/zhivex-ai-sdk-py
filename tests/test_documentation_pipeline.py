from __future__ import annotations

import hashlib
import io
import json
from unittest.mock import patch

import pytest

from scripts.build_docs import check_site, download_published
from scripts.render_docs_reference import check_imports


@pytest.mark.parametrize(
    "source",
    [
        "from zhivex_ai.providers.openai_compat import OpenAICompatibleLanguageModel",
        "import zhivex_ai.agent",
        "from zhivex_ai import invented_api",
        "from zhivex_ai import *",
    ],
)
def test_snippet_rejects_non_public_imports(source):
    with pytest.raises(ValueError, match="Unsupported"):
        check_imports(source, {"zhivex_ai": {"Agent"}}, "example.py")


def test_snippet_accepts_installed_public_names():
    check_imports(
        "from zhivex_ai import Agent\nimport zhivex_ai",
        {"zhivex_ai": {"Agent"}},
        "example.py",
    )


@pytest.mark.parametrize("href", ["missing/", "other.html#missing", "missing.js"])
def test_built_site_rejects_missing_targets(tmp_path, href):
    (tmp_path / "index.html").write_text(f'<a href="{href}">Broken</a>')
    (tmp_path / "other.html").write_text('<h1 id="present">Target</h1>')
    with pytest.raises(ValueError, match="missing"):
        check_site(tmp_path)


def test_version_prefix_and_fragment_links(tmp_path):
    (tmp_path / "index.html").write_text(
        '<a href="/sdk/0.23.0/other.html#present">OK</a>'
    )
    (tmp_path / "other.html").write_text('<h1 id="present">Target</h1>')
    check_site(tmp_path, "/sdk/0.23.0")


def test_published_artifact_rejects_tampered_bytes(tmp_path):
    expected = hashlib.sha256(b"expected").hexdigest()
    metadata = {
        "urls": [
            {
                "filename": "sdk.whl",
                "digests": {"sha256": expected},
                "url": "https://files.pythonhosted.org/sdk.whl",
            }
        ]
    }
    with patch(
        "urllib.request.urlopen",
        side_effect=[io.BytesIO(json.dumps(metadata).encode()), io.BytesIO(b"wrong")],
    ):
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            download_published(
                {"version": "0.23.0", "wheel_sha256": expected}, tmp_path
            )
