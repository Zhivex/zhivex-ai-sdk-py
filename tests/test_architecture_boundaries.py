"""Executable dependency boundaries for the portable SDK and agent internals."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "zhivex_ai"
_EXTENSION_LAYERS = {
    "cli", "skill_cli", "responses_host", "evals", "workflow", "workflow_adapters",
    "workflow_graph", "workflow_state", "workflows", "experimental", "hosting",
}
_FOUNDATION_MODULES = {
    "types", "errors", "schema", "messages", "runtime", "transport", "_http",
    "_streaming", "generate_text", "generate_object", "embed", "grounded_text",
    "protocols", "audio", "agent", "agent_state",
}


def _runtime_imports(source: str, module: str) -> set[str]:
    """Find imports at every depth, excluding only explicit typing branches."""
    found: set[str] = set()

    def is_type_checking(test: ast.expr) -> bool:
        return (
            isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
            or isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.If) and is_type_checking(node.test):
            for child in node.orelse:
                walk(child)
            return
        if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not) and is_type_checking(node.test.operand):
            for child in node.body:
                walk(child)
            return
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            package = module.rsplit(".", 1)[0]
            target = importlib.util.resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module or ""
            found.add(target)
            found.update(f"{target}.{alias.name}" for alias in node.names)
        for child in ast.iter_child_nodes(node):
            walk(child)

    walk(ast.parse(source))
    return found


def _violations(path: Path, source: str) -> set[str]:
    module = "zhivex_ai." + ".".join(path.relative_to(PACKAGE).with_suffix("").parts)
    imports = _runtime_imports(source, module)
    forbidden: set[str] = set()
    if path.stem in _FOUNDATION_MODULES or path.stem.startswith("_agent_"):
        for dependency in imports:
            if dependency.startswith("zhivex_ai.") and dependency.split(".")[1] in _EXTENSION_LAYERS:
                forbidden.add(dependency)
    if path.stem in {"_agent_contracts", "_agent_tools", "_agent_execution", "_agent_skills", "_agent_approvals", "_agent_memory", "_agent_persistence", "_agent_run_state", "_agent_streams", "_agent_live"}:
        forbidden.update(dependency for dependency in imports if dependency == "zhivex_ai.agent" or dependency.startswith("zhivex_ai.agent."))
    if path.stem == "_agent_contracts":
        forbidden.update(dependency for dependency in imports if dependency == "zhivex_ai.agent_state" or dependency.startswith("zhivex_ai.agent_state."))
    if path == PACKAGE / "providers" / "base.py":
        forbidden.update(dependency for dependency in imports if dependency.startswith("zhivex_ai.providers.") and not dependency.startswith("zhivex_ai.providers._native_extensions"))
    return forbidden


@pytest.mark.parametrize("path", sorted(
    [path for path in PACKAGE.glob("*.py") if path.stem in _FOUNDATION_MODULES or path.stem.startswith("_agent_")]
    + [PACKAGE / "providers" / "base.py"],
), ids=lambda path: path.name)
def test_runtime_dependencies_respect_architecture(path: Path) -> None:
    assert not _violations(path, path.read_text()), f"{path.name}: forbidden runtime dependencies"


def test_architecture_detector_rejects_nested_and_absolute_imports() -> None:
    path = PACKAGE / "_agent_tools.py"
    assert "zhivex_ai.agent" in _violations(path, "def f():\n    from .agent import Agent\n")
    assert "zhivex_ai.workflows" in _violations(path, "import zhivex_ai.workflows\n")
    assert "zhivex_ai.responses_host" in _violations(path, "from . import responses_host\n")
    assert not _violations(path, "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .agent import Agent\n")
    assert "zhivex_ai.agent" in _violations(path, "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    pass\nelse:\n    from .agent import Agent\n")
    assert "zhivex_ai.providers._native_sessions" in _violations(PACKAGE / "providers" / "base.py", "from ._native_sessions import OpenAILiveClient\n")


def test_minimal_sdk_import_does_not_load_optional_runtimes() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    code = '''
import sys
import zhivex_ai
from zhivex_ai import Agent, create_openai
optional = ("fastapi", "asyncpg", "mcp", "websockets", "opentelemetry")
loaded = sorted(name for name in sys.modules if any(name == prefix or name.startswith(prefix + ".") for prefix in optional))
assert not loaded, loaded
'''
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
