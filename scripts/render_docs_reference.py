"""Run with an isolated wheel's Python; never add checkout src to sys.path."""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

NAMESPACES = {
    "root": "zhivex_ai",
    "workflows": "zhivex_ai.workflows",
    "evals": "zhivex_ai.evals",
    "protocols": "zhivex_ai.integrations.protocols",
    "responses": "zhivex_ai.integrations.responses",
    "experimental": "zhivex_ai.experimental",
    "experimental-providers": "zhivex_ai.experimental.providers",
    "experimental-realtime": "zhivex_ai.experimental.realtime",
}


def check_imports(source: str, exports: dict[str, set[str]], filename: str) -> None:
    """Reject deep imports and names absent from the installed public contract."""
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "zhivex_ai"
        ):
            if node.module not in exports or any(
                a.name not in exports[node.module] for a in node.names
            ):
                raise ValueError(
                    f"Unsupported public import in {filename}:{node.lineno}"
                )
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("zhivex_ai") and alias.name not in exports:
                    raise ValueError(
                        f"Unsupported namespace in {filename}:{node.lineno}"
                    )
    compile(tree, filename, "exec")


def render(
    source_dir: Path, output: Path, metadata: dict[str, str]
) -> dict[str, object]:
    import zhivex_ai
    from zhivex_ai.api_stability import API_STABILITY

    actual = version("zhivex-ai-sdk")
    if actual != metadata["version"]:
        raise ValueError("Installed wheel version differs from documentation identity")
    package_file = Path(zhivex_ai.__file__).resolve()
    if "site-packages" not in package_file.parts:
        raise ValueError(
            "Documentation requires an installed wheel, not checkout imports"
        )
    modules = {slug: importlib.import_module(name) for slug, name in NAMESPACES.items()}
    exports = {m.__name__: set(m.__all__) for m in modules.values()}
    if exports["zhivex_ai"] != set(API_STABILITY):
        raise ValueError("Public exports and stability manifest diverge")
    output.mkdir(parents=True, exist_ok=True)
    reference = output / "reference"
    reference.mkdir(exist_ok=True)
    api_entries = []
    stable_documented = set()
    for slug, module in modules.items():
        parts = [
            f"# {module.__name__}",
            "Public import paths and stability come from the installed artifact.",
            "Use the root for Stable APIs and focused namespaces for extensions. Legacy root extension imports remain classified below.",
        ]
        for name in sorted(module.__all__):
            entry = API_STABILITY.get(name)
            if entry is None:
                raise ValueError(
                    f"Missing stability classification: {module.__name__}.{name}"
                )
            obj = getattr(module, name)
            try:
                signature = str(inspect.signature(obj))
            except (TypeError, ValueError):
                signature = " (public type alias or constant; no callable signature)"
            signature = re.sub(r" at 0x[0-9a-fA-F]+", "", signature)
            parts.extend(
                [
                    f"## {name}",
                    f"**{entry.level.title()}** · {entry.category}",
                    f"```python\nfrom {module.__name__} import {name}\n```",
                    f"```text\n{name}{signature}\n```",
                ]
            )
            doc = (
                inspect.getdoc(obj)
                if inspect.isfunction(obj) or inspect.isclass(obj)
                else None
            )
            if doc:
                parts.append("```text\n" + doc.replace("```", "'''") + "\n```")
            if entry.notes:
                parts.append(entry.notes)
            api_entries.append(
                {"namespace": module.__name__, "name": name, "level": entry.level}
            )
            if module.__name__ == "zhivex_ai" and entry.level == "stable":
                stable_documented.add(name)
        (reference / f"{slug}.md").write_text("\n\n".join(parts) + "\n")
    expected_stable = {
        name for name, entry in API_STABILITY.items() if entry.level == "stable"
    }
    if stable_documented != expected_stable:
        raise ValueError("Incomplete Stable API reference")
    snippet_policy = json.loads((source_dir / "snippets.json").read_text())
    snippet_files = {
        str(p.relative_to(source_dir)) for p in (source_dir / "snippets").glob("*.py")
    }
    if set(snippet_policy) != snippet_files:
        raise ValueError("Every snippet must have an explicit execution policy")
    for relative, mode in snippet_policy.items():
        path = source_dir / relative
        code = path.read_text()
        check_imports(code, exports, relative)
        if mode == "execute":
            subprocess.run(
                [sys.executable, "-I", str(path.resolve())],
                check=True,
                timeout=30,
                cwd=output,
                capture_output=True,
            )
        elif mode != "compile":
            raise ValueError(f"Unknown snippet mode: {mode}")
    source_url = (
        "https://github.com/Zhivex/zhivex-ai-sdk-py/blob/" + metadata["source_commit"]
    )
    compiled_blocks = 0
    for path in sorted(source_dir.glob("*.md")):
        content = (
            path.read_text()
            .replace("{{VERSION}}", actual)
            .replace("{{SOURCE}}", source_url)
        )

        def include(match: re.Match[str]) -> str:
            relative = match.group(1)
            if relative not in snippet_policy:
                raise ValueError(f"Unregistered snippet: {relative}")
            return (
                "```python\n" + (source_dir / relative).read_text().rstrip() + "\n```"
            )

        content = re.sub(r"<!-- snippet: ([\w/.-]+) -->", include, content)
        for block in re.findall(r"```python\n(.*?)```", content, re.S):
            check_imports(block, exports, str(path.name))
            compiled_blocks += 1
        (output / path.name).write_text(content)
    identity = (
        f"\n\n## Documentation artifact\n\nVersion **{actual}** · {metadata['channel']} · package maturity **Beta**.\n\n"
        f"Wheel SHA256: `{metadata['wheel_sha256']}`. Source: `{metadata['source_commit']}`.\n\n"
        "These identifiers describe the documented package; site changes do not publish a new SDK release.\n"
    )
    index = output / "index.md"
    index.write_text(index.read_text() + identity)
    (reference / "index.md").write_text(
        "# API reference\n\n"
        + "\n".join(f"- [{name}]({slug}.md)" for slug, name in NAMESPACES.items())
        + identity
    )
    report = {
        **metadata,
        "stable_total": len(expected_stable),
        "stable_documented": len(stable_documented),
        "api_entries": api_entries,
        "snippets": snippet_policy,
        "compiled_blocks": compiled_blocks,
        "installed_wheel": True,
    }
    (output / "documentation-evidence.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    report = render(args.source, args.output, json.loads(args.metadata.read_text()))
    print(json.dumps({k: v for k, v in report.items() if k != "api_entries"}))
