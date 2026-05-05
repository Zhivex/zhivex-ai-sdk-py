from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import venv


ROOT = Path(__file__).resolve().parents[1]
EXTRAS_IMPORTS = {
    "postgres": ["asyncpg"],
    "mcp": ["mcp"],
    "api": ["fastapi", "uvicorn"],
    "otel": ["opentelemetry", "opentelemetry.sdk"],
    "docx": ["docx"],
}


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True, env=env)


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_bin(venv_dir: Path, name: str) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def _create_venv(path: Path) -> Path:
    builder = venv.EnvBuilder(with_pip=True, clear=True)
    try:
        builder.create(path)
    except subprocess.CalledProcessError:
        uv = shutil.which("uv")
        if uv is None:
            raise
        _run([uv, "venv", "--seed", str(path)])
    return _venv_python(path)


def _latest_artifact(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern), key=lambda item: item.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No release artifact matched {pattern!r} in {dist_dir}.")
    return matches[-1]


def _install(python: Path, requirement: str) -> None:
    _run([str(python), "-m", "pip", "install", "--disable-pip-version-check", requirement])


def _smoke_code() -> str:
    return textwrap.dedent(
        """
        import asyncio
        from importlib import metadata, resources

        import zhivex_ai
        from zhivex_ai import (
            Agent,
            SequentialAgent,
            WorkflowStep,
            create_mock_language_model,
            generate_text,
            run_agent,
        )
        from zhivex_ai.types import GenerateResult

        assert metadata.version("zhivex-ai-sdk")
        assert "Agent" in zhivex_ai.__all__
        assert "generate_text" in zhivex_ai.__all__
        assert resources.files("zhivex_ai").joinpath("py.typed").is_file()

        async def main():
            model = create_mock_language_model(responses=[GenerateResult(text="hello", finish_reason="stop")])
            text = await generate_text(model=model, prompt="hello")
            assert text.text == "hello"

            agent = Agent(
                name="assistant",
                model=create_mock_language_model(responses=[GenerateResult(text="agent-ok", finish_reason="stop")]),
            )
            run = await run_agent(agent=agent, prompt="go")
            assert run.text == "agent-ok"

            workflow = SequentialAgent(
                name="release_smoke",
                steps=[
                    WorkflowStep(
                        "step",
                        Agent(
                            name="worker",
                            model=create_mock_language_model(
                                responses=[GenerateResult(text="workflow-ok", finish_reason="stop")]
                            ),
                        ),
                        prompt="go",
                        output_key="out",
                    )
                ],
            )
            result = await workflow.run()
            assert result.state["out"] == "workflow-ok"

        asyncio.run(main())
        """
    )


def _run_base_smoke(python: Path) -> None:
    _run([str(python), "-c", _smoke_code()])
    cli = _venv_bin(python.parent.parent, "zhivex-skills")
    _run([str(cli), "--help"])


def _run_extra_smoke(python: Path, extra: str) -> None:
    imports = EXTRAS_IMPORTS[extra]
    code = "\n".join(
        [
            "import importlib",
            "import zhivex_ai",
            *[f'importlib.import_module("{name}")' for name in imports],
            'print("ok")',
        ]
    )
    _run([str(python), "-c", code])


def verify_wheel(wheel: Path, *, extras: list[str]) -> None:
    with tempfile.TemporaryDirectory(prefix="zhivex-wheel-smoke-") as temp_dir:
        python = _create_venv(Path(temp_dir) / "venv")
        _install(python, str(wheel))
        _run_base_smoke(python)

    for extra in extras:
        with tempfile.TemporaryDirectory(prefix=f"zhivex-extra-{extra}-") as temp_dir:
            python = _create_venv(Path(temp_dir) / "venv")
            requirement = f"zhivex-ai-sdk[{extra}] @ {wheel.resolve().as_uri()}"
            _install(python, requirement)
            _run_extra_smoke(python, extra)


def verify_sdist(sdist: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="zhivex-sdist-smoke-") as temp_dir:
        python = _create_venv(Path(temp_dir) / "venv")
        _install(python, str(sdist))
        _run_base_smoke(python)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify built Zhivex AI SDK release artifacts in fresh venvs.")
    parser.add_argument("--dist-dir", default=str(ROOT / "dist"))
    parser.add_argument("--skip-sdist", action="store_true", help="Skip installing the source distribution.")
    parser.add_argument(
        "--extras",
        default="postgres,mcp,api,otel,docx",
        help="Comma-separated optional extras to install and import-check.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dist_dir = Path(args.dist_dir)
    wheel = _latest_artifact(dist_dir, "zhivex_ai_sdk-*.whl")
    sdist = _latest_artifact(dist_dir, "zhivex_ai_sdk-*.tar.gz")
    extras = [item.strip() for item in args.extras.split(",") if item.strip()]
    unknown = sorted(set(extras) - set(EXTRAS_IMPORTS))
    if unknown:
        raise ValueError(f"Unknown extras requested: {', '.join(unknown)}")

    env_hint = os.getenv("ZHIVEX_RELEASE_VERIFY_NETWORK")
    if env_hint:
        print(f"ZHIVEX_RELEASE_VERIFY_NETWORK={env_hint}")

    verify_wheel(wheel, extras=extras)
    if not args.skip_sdist:
        verify_sdist(sdist)
    print("Release artifacts verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
