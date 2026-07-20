from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class ReleaseCheck:
    name: str
    command: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True, slots=True)
class ReleaseCheckResult:
    name: str
    command: tuple[str, ...]
    status: str
    output: str


@dataclass(frozen=True, slots=True)
class ReleaseMetadata:
    git_commit: str
    working_tree: str
    python_version: str
    tool_versions: tuple[tuple[str, str], ...]
    artifact_hashes: tuple[tuple[str, str], ...]


DEFAULT_CHECKS = (
    ReleaseCheck("compile", ("make", "compile")),
    ReleaseCheck("lint", ("make", "lint")),
    ReleaseCheck("typecheck", ("make", "typecheck")),
    ReleaseCheck("support matrix", ("make", "support-matrix-check")),
    ReleaseCheck("public contract", ("make", "test-contract")),
    ReleaseCheck("core runtime", ("make", "test-core")),
    ReleaseCheck("providers", ("make", "test-providers")),
    ReleaseCheck("agents and workflows", ("make", "test-agents")),
    ReleaseCheck("examples", ("make", "test-examples")),
    ReleaseCheck("release tooling", ("make", "test-release")),
    ReleaseCheck("build artifacts", ("make", "build")),
    ReleaseCheck("artifact install smoke", ("make", "release-install-check")),
    ReleaseCheck("dependency audit", ("make", "security-check")),
)


def _package_version() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    return str(pyproject["project"]["version"])


def _run_check(check: ReleaseCheck, *, dry_run: bool) -> ReleaseCheckResult:
    if dry_run:
        return ReleaseCheckResult(check.name, check.command, "not-run", "Dry run; command was not executed.")
    process = subprocess.run(
        check.command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    status = "passed" if process.returncode == 0 else "failed"
    output = "\n".join(line.rstrip() for line in process.stdout.strip().splitlines())
    return ReleaseCheckResult(check.name, check.command, status, output)


def _git_output(*args: str) -> str | None:
    process = subprocess.run(
        ("git", *args),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if process.returncode != 0:
        return None
    return process.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_release_metadata() -> ReleaseMetadata:
    git_commit = _git_output("rev-parse", "HEAD") or "unavailable"
    status_output = _git_output("status", "--porcelain", "--untracked-files=normal")
    working_tree = "unavailable" if status_output is None else ("dirty" if status_output else "clean")
    python_version = sys.version.split()[0]
    tool_versions: list[tuple[str, str]] = []
    for package in ("build", "hatchling", "pip-audit", "twine"):
        try:
            version = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            version = "not-installed"
        tool_versions.append((package, version))

    dist_dir = ROOT / "dist"
    artifact_prefix = f"zhivex_ai_sdk-{_package_version().replace('-', '_')}"
    artifact_hashes = tuple(
        (path.name, _sha256_file(path))
        for path in sorted(dist_dir.iterdir() if dist_dir.is_dir() else ())
        if path.is_file()
        and path.name.startswith(artifact_prefix)
        and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )
    return ReleaseMetadata(
        git_commit=git_commit,
        working_tree=working_tree,
        python_version=python_version,
        tool_versions=tuple(tool_versions),
        artifact_hashes=artifact_hashes,
    )


def _render_markdown(
    version: str,
    results: list[ReleaseCheckResult],
    *,
    dry_run: bool,
    metadata: ReleaseMetadata,
) -> str:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines = [
        f"# Zhivex AI SDK {version} Release Evidence",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Mode: `{'dry-run' if dry_run else 'executed'}`",
        "",
        "## Candidate Metadata",
        "",
        f"- Git commit: `{metadata.git_commit}`",
        f"- Working tree: `{metadata.working_tree}`",
        f"- Python: `{metadata.python_version}`",
        "- Tool versions:",
    ]
    for package, tool_version in metadata.tool_versions:
        lines.append(f"  - `{package}=={tool_version}`")
    lines.extend(
        [
            "",
            "## Artifact SHA256",
            "",
        ]
    )
    if metadata.artifact_hashes:
        for filename, digest in metadata.artifact_hashes:
            lines.append(f"- `{filename}`: `{digest}`")
    else:
        lines.append("- No wheel or source distribution was present in `dist/`.")
    lines.extend(
        [
            "",
            "## Gate Summary",
            "",
            "| Gate | Status | Command |",
            "| --- | --- | --- |",
        ]
    )
    for result in results:
        lines.append(f"| {result.name} | {result.status} | `{' '.join(result.command)}` |")
    lines.extend(["", "## Output"])
    for result in results:
        lines.extend(
            [
                "",
                f"### {result.name}",
                "",
                f"Command: `{' '.join(result.command)}`",
                "",
                "```text",
                result.output or "(no output)",
                "```",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and record Zhivex AI SDK release evidence gates.")
    parser.add_argument("--version", default=_package_version())
    parser.add_argument(
        "--output",
        default=None,
        help="Markdown output path. Defaults to docs/releases/<version>-evidence.md.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Render the evidence file without running gates.")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated gate names to run, using the names shown in the evidence table.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected_names = {item.strip() for item in args.only.split(",")} if args.only else None
    checks = [check for check in DEFAULT_CHECKS if selected_names is None or check.name in selected_names]
    if selected_names:
        missing = sorted(selected_names - {check.name for check in DEFAULT_CHECKS})
        if missing:
            raise ValueError(f"Unknown release evidence gates: {', '.join(missing)}")

    results = [_run_check(check, dry_run=args.dry_run) for check in checks]
    metadata = _collect_release_metadata()
    output = Path(args.output) if args.output else ROOT / "docs" / "releases" / f"{args.version}-evidence.md"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _render_markdown(args.version, results, dry_run=args.dry_run, metadata=metadata),
        "utf-8",
    )

    failed = [result for result in results if result.status == "failed"]
    try:
        display_path = output.relative_to(ROOT)
    except ValueError:
        display_path = output
    print(f"Wrote release evidence to {display_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
