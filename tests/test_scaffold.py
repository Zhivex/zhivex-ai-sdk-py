from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from zhivex_ai.cli import main
from zhivex_ai._scaffold import create_project


@pytest.mark.parametrize(
    "name", ["../escape", "/tmp/escape", ".", "..", "a/b", "a\\b", "", "-name"]
)
def test_scaffold_rejects_escaping_names(tmp_path, name):
    with patch("pathlib.Path.cwd", return_value=tmp_path), pytest.raises(ValueError):
        create_project(name, backend="sqlite", sdk_version="0.23.0")
    assert list(tmp_path.iterdir()) == []


def test_scaffold_preserves_existing_files_and_symlinks(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    sentinel = target / "keep"
    sentinel.write_text("keep")
    (tmp_path / "linked").symlink_to(target)
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        for name in ["existing", "linked"]:
            with pytest.raises(FileExistsError):
                create_project(name, backend="sqlite", sdk_version="0.23.0")
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_scaffold_contract(tmp_path, backend):
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        assert main(["init", "demo", "--backend", backend]) == 0
    root = tmp_path / "demo"
    assert {p.name for p in root.iterdir()} == {
        "app.py",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "README.md",
        "test_app.py",
        "pyproject.toml",
        "scaffold.json",
    }
    source = (root / "app.py").read_text()
    compile(source, "app.py", "exec")
    assert "sys.path" not in source and "__BACKEND__" not in source
    assert not (root / ".env").exists()
    assert all(
        line.endswith("=") for line in (root / ".env.example").read_text().splitlines()
    )
    if os.name == "posix":
        assert root.stat().st_mode & 0o777 == 0o700


def test_generated_tool_schema_is_closed_for_strict_providers(tmp_path):
    import importlib.util
    import sys
    with patch('pathlib.Path.cwd', return_value=tmp_path):
        root = create_project('demo', backend='sqlite', sdk_version='0.23.0')
    spec = importlib.util.spec_from_file_location('generated_schema_test', root / 'app.py')
    app = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'generated_schema_test': app}):
        spec.loader.exec_module(app)
        schema = app.ProjectInput.model_json_schema()
        assert schema['additionalProperties'] is False
        assert set(schema['required']) == set(schema['properties'])
