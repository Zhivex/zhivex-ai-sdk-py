from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "zhivex-ai-sdk"


def _compile_command(*, uv: str, requirements: Path, cache_dir: Path) -> list[str]:
    return [
        uv,
        "pip",
        "compile",
        str(ROOT / "pyproject.toml"),
        "--all-extras",
        "--universal",
        "--generate-hashes",
        "--no-annotate",
        "--no-header",
        "--quiet",
        "--no-emit-package",
        PROJECT_NAME,
        "--cache-dir",
        str(cache_dir),
        "--output-file",
        str(requirements),
    ]


def _audit_command(*, requirements: Path, cache_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pip_audit",
        "--requirement",
        str(requirements),
        "--no-deps",
        "--require-hashes",
        "--disable-pip",
        "--cache-dir",
        str(cache_dir),
        "--progress-spinner",
        "off",
    ]


def _run(command: list[str], *, quiet: bool = False) -> None:
    print("+", " ".join(command), flush=True)
    if not quiet:
        subprocess.run(command, cwd=ROOT, check=True)
        return

    try:
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, file=sys.stderr)
        raise


def main() -> int:
    uv = shutil.which("uv")
    if uv is None:
        print("Dependency audit requires the uv executable.", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="zhivex-dependency-audit-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        requirements = temporary_path / "all-extras-requirements.txt"
        _run(
            _compile_command(
                uv=uv,
                requirements=requirements,
                cache_dir=temporary_path / "uv-cache",
            ),
            quiet=True,
        )
        _run(
            _audit_command(
                requirements=requirements,
                cache_dir=temporary_path / "pip-audit-cache",
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
